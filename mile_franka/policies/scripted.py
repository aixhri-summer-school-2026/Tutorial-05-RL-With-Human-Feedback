"""Scripted block-stacking policy over ground-truth cube poses (spec sec 5.5).

A state machine: approach -> pregrasp -> descend -> grasp -> lift -> over-base
-> preplace -> place -> release.
Reads one FrankaEnv frame (18,) and returns a 4-DoF delta action in [-1, 1]. With
mediocre=True it injects an aim bias, a per-episode random xy offset, action noise, and a
release-height error so it starts the task but cannot reliably finish -- the regime MILE
learns from. The skilled variant (mediocre=False) is the expert the ScriptedIntervener uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mile_franka.config import StackTaskConfig

# Indices into one FrankaEnv frame (18,): see Phase 2 observation convention.
EE = slice(0, 3)
GRIPPER_W = 3
TOP_POS = slice(4, 7)
BOTTOM_POS = slice(11, 14)

# State-machine phases.
(
    _APPROACH,
    _PREGRASP,
    _DESCEND,
    _GRASP,
    _LIFT,
    _OVER_BASE,
    _PREPLACE,
    _PLACE,
    _RELEASE,
    _DONE,
) = range(10)


@dataclass
class ScriptedPolicyConfig:
    hover_height: float = 0.30        # m above the table for transit
    pregrasp_clearance: float = 0.050  # EE clearance above cube top before vertical descent
    preplace_clearance: float = 0.050  # EE clearance above stack target before placing
    grasp_center_offset: float = 0.005  # EE height above cube center while closing
    pos_tol: float = 0.018            # default within-target tolerance
    transit_pos_tol: float = 0.045    # loose tolerance for high, collision-free moves
    transit_xy_tol: float = 0.030     # high waypoints only need coarse xy alignment
    pregrasp_xy_tol: float = 0.005    # centered over cube before descending
    pregrasp_z_tol: float = 0.025     # close to the low pregrasp height
    grasp_xy_tol: float = 0.008       # centered before closing gripper
    grasp_z_tol: float = 0.020        # close enough vertically to close gripper
    pre_place_xy_tol: float = 0.012   # centered above stack before descending
    pre_place_z_tol: float = 0.025    # close to the low preplace height
    place_xy_tol: float = 0.012       # release only when horizontally centered
    place_z_tol: float = 0.010        # release only when seated within success tolerance
    place_down_bias: float = 0.030    # downward preload while placing
    dwell_steps: int = 5              # steps to hold while grasping/releasing
    waypoint_step: float = 0.035      # max Cartesian waypoint advance per 10 Hz tick
    descent_waypoint_step: float = 0.010  # slower vertical insertion toward grasp/place
    slow_radius: float = 0.09         # begin slowing this far from a phase goal
    slow_fraction: float = 0.45       # waypoint fraction of remaining error near goal
    min_waypoint_step: float = 0.006  # keep enough progress to avoid static waits
    max_phase_steps: int = 80         # close-enough guard against controller steady error
    # Mediocre knobs (ignored when mediocre=False):
    aim_xy_bias: float = 0.0          # constant horizontal placement bias (m)
    aim_xy_noise_std: float = 0.015   # per-episode random horizontal offset std (m)
    action_noise_std: float = 0.04    # gaussian noise on the [-1,1] action
    release_height_error: float = 0.010  # release this much too high (m)


class ScriptedStackPolicy:
    """Deterministic (optionally mediocre) cube stacker. Stateful: call reset() per episode."""

    def __init__(self, task: Optional[StackTaskConfig] = None,
                 cfg: Optional[ScriptedPolicyConfig] = None, mediocre: bool = True):
        self.task = task if task is not None else StackTaskConfig()
        self.cfg = cfg if cfg is not None else ScriptedPolicyConfig()
        self.mediocre = mediocre
        self._phase = _APPROACH
        self._dwell = 0
        self._episode_offset = np.zeros(2, dtype=np.float32)
        self._grasp_offset = np.zeros(3, dtype=np.float32)
        self._grasp_top = np.zeros(3, dtype=np.float32)
        self._place_bottom = np.zeros(3, dtype=np.float32)
        self._phase_steps = 0
        self._rng = np.random.default_rng()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self._phase = _APPROACH
        self._dwell = 0
        self._grasp_offset = np.zeros(3, dtype=np.float32)
        self._grasp_top = np.zeros(3, dtype=np.float32)
        self._place_bottom = np.zeros(3, dtype=np.float32)
        self._phase_steps = 0
        if rng is not None:
            self._rng = rng
        if self.mediocre:
            self._episode_offset = (
                np.array([self.cfg.aim_xy_bias, self.cfg.aim_xy_bias], dtype=np.float32)
                + self._rng.normal(0.0, self.cfg.aim_xy_noise_std, size=2).astype(np.float32))
        else:
            self._episode_offset = np.zeros(2, dtype=np.float32)

    def _delta_to_pose(self, ee: np.ndarray, command_pose: np.ndarray) -> np.ndarray:
        return np.clip((command_pose - ee) / self.task.action_scale, -1.0, 1.0)

    def _next_waypoint(self, ee: np.ndarray, target: np.ndarray,
                       slow_near_goal: bool = False,
                       waypoint_step: Optional[float] = None) -> np.ndarray:
        delta = target - ee
        dist = float(np.linalg.norm(delta))
        step = self.cfg.waypoint_step if waypoint_step is None else waypoint_step
        if slow_near_goal and dist < self.cfg.slow_radius:
            step = min(step, max(self.cfg.min_waypoint_step, dist * self.cfg.slow_fraction))
        if dist <= step:
            return target
        return (ee + delta / dist * step).astype(np.float32)

    def _reached(self, ee: np.ndarray, target: np.ndarray,
                 tol: Optional[float] = None) -> bool:
        tol = self.cfg.pos_tol if tol is None else tol
        return bool(np.linalg.norm(target - ee) < tol)

    def _reached_xy_z(self, ee: np.ndarray, target: np.ndarray,
                      xy_tol: float, z_tol: float) -> bool:
        xy_ok = np.linalg.norm(target[:2] - ee[:2]) < xy_tol
        z_ok = abs(float(target[2] - ee[2])) < z_tol
        return bool(xy_ok and z_ok)

    def _set_phase(self, phase: int) -> None:
        if self._phase != phase:
            self._phase = phase
            self._phase_steps = 0

    def _phase_timed_out(self) -> bool:
        return self._phase_steps >= self.cfg.max_phase_steps

    def act(self, frame: np.ndarray) -> np.ndarray:
        """frame: a single FrankaEnv observation (18,). Returns action (4,) float32."""
        ee = np.asarray(frame[EE], dtype=np.float32)
        top = np.asarray(frame[TOP_POS], dtype=np.float32)
        bottom = np.asarray(frame[BOTTOM_POS], dtype=np.float32)
        hover_z = self.task.table_z + self.cfg.hover_height
        cube_top_z = top[2] + 0.5 * self.task.cube_size
        pregrasp_z = cube_top_z + self.cfg.pregrasp_clearance
        grasp_top = self._grasp_top if self._phase in (_DESCEND, _GRASP, _LIFT) else top
        grasp_z = grasp_top[2] + self.cfg.grasp_center_offset
        place_bottom = self._place_bottom if self._phase in (
            _PREPLACE, _PLACE, _RELEASE, _DONE) else bottom
        place_xy = place_bottom[:2] + (self._episode_offset if self.mediocre else 0.0)
        place_z = place_bottom[2] + self.task.cube_size + (
            self.cfg.release_height_error if self.mediocre else 0.0)
        top_place_target = np.array([place_xy[0], place_xy[1], place_z], dtype=np.float32)
        preplace_z = place_z + self.cfg.preplace_clearance + self._grasp_offset[2]

        gripper = -1.0  # default open
        phase_tol = self.cfg.pos_tol
        command_phase = self._phase
        self._phase_steps += 1
        if self._phase == _APPROACH:
            phase_tol = self.cfg.transit_pos_tol
            target = np.array([top[0], top[1], hover_z], dtype=np.float32)
            if (self._reached_xy_z(ee, target, self.cfg.transit_xy_tol, phase_tol)
                    or self._phase_timed_out()):
                self._set_phase(_PREGRASP)
        elif self._phase == _PREGRASP:
            phase_tol = self.cfg.pregrasp_xy_tol
            target = np.array([top[0], top[1], pregrasp_z], dtype=np.float32)
            if (self._reached_xy_z(
                    ee, target, self.cfg.pregrasp_xy_tol, self.cfg.pregrasp_z_tol)
                    or (self._phase_timed_out()
                        and self._reached_xy_z(ee, target, 0.008, 0.050))):
                self._grasp_top = top.copy()
                self._set_phase(_DESCEND)
        elif self._phase == _DESCEND:
            phase_tol = self.cfg.grasp_xy_tol
            target = np.array(
                [grasp_top[0], grasp_top[1],
                 max(grasp_z, self.task.table_z + self.cfg.grasp_center_offset)],
                dtype=np.float32,
            )
            if (self._reached_xy_z(ee, target, self.cfg.grasp_xy_tol, self.cfg.grasp_z_tol)
                    or (self._phase_timed_out()
                        and self._reached_xy_z(ee, target, 0.012, 0.035))):
                self._set_phase(_GRASP)
        elif self._phase == _GRASP:
            target, gripper = ee.copy(), 1.0
            self._dwell += 1
            if self._dwell >= self.cfg.dwell_steps:
                self._grasp_offset = (ee - top).astype(np.float32)
                self._dwell = 0
                self._set_phase(_LIFT)
        elif self._phase == _LIFT:
            phase_tol = self.cfg.transit_pos_tol
            self._grasp_offset = (ee - top).astype(np.float32)
            target, gripper = np.array([ee[0], ee[1], hover_z], dtype=np.float32), 1.0
            if self._reached(ee, target, phase_tol) or self._phase_timed_out():
                self._set_phase(_OVER_BASE)
        elif self._phase == _OVER_BASE:
            phase_tol = self.cfg.transit_pos_tol
            self._grasp_offset = (ee - top).astype(np.float32)
            target, gripper = np.array([place_xy[0], place_xy[1], hover_z], dtype=np.float32), 1.0
            if (self._reached_xy_z(ee, target, self.cfg.transit_xy_tol, phase_tol)
                    or self._phase_timed_out()):
                self._place_bottom = bottom.copy()
                self._grasp_offset = (ee - top).astype(np.float32)
                self._set_phase(_PREPLACE)
        elif self._phase == _PREPLACE:
            phase_tol = self.cfg.pre_place_xy_tol
            self._grasp_offset = (ee - top).astype(np.float32)
            target = np.array([place_xy[0], place_xy[1], preplace_z], dtype=np.float32)
            gripper = 1.0
            if (self._reached_xy_z(
                    ee, target, self.cfg.pre_place_xy_tol, self.cfg.pre_place_z_tol)
                    or (self._phase_timed_out()
                        and self._reached_xy_z(ee, target, 0.020, 0.050))):
                self._set_phase(_PLACE)
        elif self._phase == _PLACE:
            phase_tol = self.cfg.place_xy_tol
            target, gripper = top_place_target + self._grasp_offset, 1.0
            target = target.copy()
            target[2] -= self.cfg.place_down_bias
            if self._reached_xy_z(top, top_place_target,
                                  self.cfg.place_xy_tol, self.cfg.place_z_tol):
                self._set_phase(_RELEASE)
        elif self._phase == _RELEASE:
            target, gripper = ee.copy(), -1.0
            self._dwell += 1
            if self._dwell >= self.cfg.dwell_steps:
                self._dwell = 0
                self._set_phase(_DONE)
        else:  # _DONE
            target, gripper = ee.copy(), -1.0

        command_pose = self._next_waypoint(
            ee, target,
            slow_near_goal=command_phase in (_DESCEND, _PREPLACE, _PLACE),
            waypoint_step=(
                self.cfg.descent_waypoint_step
                if command_phase in (_DESCEND, _PREPLACE, _PLACE)
                else None
            ),
        )
        delta = self._delta_to_pose(ee, command_pose)
        if self.mediocre and self.cfg.action_noise_std > 0:
            delta = np.clip(
                delta + self._rng.normal(0.0, self.cfg.action_noise_std, size=3), -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], gripper], dtype=np.float32)
