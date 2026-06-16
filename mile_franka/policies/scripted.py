"""Scripted block-stacking policy over ground-truth cube poses (spec sec 5.5).

A state machine: approach -> descend -> grasp -> lift -> over-base -> place -> release.
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
_APPROACH, _DESCEND, _GRASP, _LIFT, _OVER_BASE, _PLACE, _RELEASE, _DONE = range(8)


@dataclass
class ScriptedPolicyConfig:
    hover_height: float = 0.20        # m above the table for transit
    pos_tol: float = 0.012            # within this of a sub-target -> advance phase
    dwell_steps: int = 3              # steps to hold while grasping/releasing
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
        self._rng = np.random.default_rng()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self._phase = _APPROACH
        self._dwell = 0
        if rng is not None:
            self._rng = rng
        if self.mediocre:
            self._episode_offset = (
                np.array([self.cfg.aim_xy_bias, self.cfg.aim_xy_bias], dtype=np.float32)
                + self._rng.normal(0.0, self.cfg.aim_xy_noise_std, size=2).astype(np.float32))
        else:
            self._episode_offset = np.zeros(2, dtype=np.float32)

    def _delta(self, ee: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.clip((target - ee) / self.task.action_scale, -1.0, 1.0)

    def _reached(self, ee: np.ndarray, target: np.ndarray) -> bool:
        return bool(np.linalg.norm(target - ee) < self.cfg.pos_tol)

    def act(self, frame: np.ndarray) -> np.ndarray:
        """frame: a single FrankaEnv observation (18,). Returns action (4,) float32."""
        ee = np.asarray(frame[EE], dtype=np.float32)
        top = np.asarray(frame[TOP_POS], dtype=np.float32)
        bottom = np.asarray(frame[BOTTOM_POS], dtype=np.float32)
        hover_z = self.task.table_z + self.cfg.hover_height
        place_xy = bottom[:2] + (self._episode_offset if self.mediocre else 0.0)
        place_z = bottom[2] + self.task.cube_size + (
            self.cfg.release_height_error if self.mediocre else 0.0)

        gripper = -1.0  # default open
        if self._phase == _APPROACH:
            target = np.array([top[0], top[1], hover_z], dtype=np.float32)
            if self._reached(ee, target):
                self._phase = _DESCEND
        elif self._phase == _DESCEND:
            target = top.copy()
            if self._reached(ee, target):
                self._phase = _GRASP
        elif self._phase == _GRASP:
            target, gripper = ee.copy(), 1.0
            self._dwell += 1
            if self._dwell >= self.cfg.dwell_steps:
                self._dwell, self._phase = 0, _LIFT
        elif self._phase == _LIFT:
            target, gripper = np.array([ee[0], ee[1], hover_z], dtype=np.float32), 1.0
            if self._reached(ee, target):
                self._phase = _OVER_BASE
        elif self._phase == _OVER_BASE:
            target, gripper = np.array([place_xy[0], place_xy[1], hover_z], dtype=np.float32), 1.0
            if self._reached(ee, target):
                self._phase = _PLACE
        elif self._phase == _PLACE:
            target, gripper = np.array([place_xy[0], place_xy[1], place_z], dtype=np.float32), 1.0
            if self._reached(ee, target):
                self._phase = _RELEASE
        elif self._phase == _RELEASE:
            target, gripper = ee.copy(), -1.0
            self._dwell += 1
            if self._dwell >= self.cfg.dwell_steps:
                self._dwell, self._phase = 0, _DONE
        else:  # _DONE
            target, gripper = ee.copy(), -1.0

        delta = self._delta(ee, target)
        if self.mediocre and self.cfg.action_noise_std > 0:
            delta = np.clip(
                delta + self._rng.normal(0.0, self.cfg.action_noise_std, size=3), -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], gripper], dtype=np.float32)
