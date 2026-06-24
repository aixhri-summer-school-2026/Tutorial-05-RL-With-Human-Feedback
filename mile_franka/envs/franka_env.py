from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from mile_franka.config import DOWN_QUAT, StackTaskConfig
from mile_franka.envs.backend import RobotBackend
from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE
from mile_franka.pose.base import ObjectPoseSource


class FrankaEnv(gym.Env):
    """Block-stacking environment over an injected RobotBackend + ObjectPoseSource."""

    metadata = {"render_modes": []}

    def __init__(self, backend: RobotBackend, pose_source: ObjectPoseSource,
                 config: Optional[StackTaskConfig] = None, mode: str = "sim",
                 obs_include_bottom_z: bool = False, z_canonicalizer=None):
        super().__init__()
        self.backend = backend
        self.pose_source = pose_source
        self.config = config if config is not None else StackTaskConfig()
        self.mode = mode
        # Reduced observation: quats and (by default) bottom_z are dropped because they carry
        # no task signal — cubes don't rotate and the bottom cube's height is the (canonical)
        # table plane. obs_include_bottom_z keeps bottom_z for the 10-dim variant.
        self.obs_include_bottom_z = obs_include_bottom_z
        # Real-only vertical canonicalizer (TablePlaneCanonicalizer) mapping ee_z/top_z into the
        # training table frame; None on sim/fake (identity), keeping the env backend-agnostic.
        self.z_canonicalizer = z_canonicalizer

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = 10 if obs_include_bottom_z else 9
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._ee_target = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._success_stable_count = 0
        self._last_success_top_pos: Optional[np.ndarray] = None
        self._last_success_pos_delta = float("inf")

    def privileged_frame(self) -> np.ndarray:
        """Full 18-dim ground-truth frame ``[ee(3), grip(1), top_pose(7), bottom_pose(7)]``.

        Scripted policies/interveners plan over full GT poses (incl. bottom_z and the cube
        quats), which the reduced policy observation drops. They read this instead of the obs
        so the BC observation can shrink without starving the scripted planner.
        """
        ee = np.asarray(self.backend.get_ee_position(), dtype=np.float32)
        width = np.array([self.backend.get_gripper_width()], dtype=np.float32)
        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        return np.concatenate([ee, width, top, bottom]).astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        ee = np.asarray(self.backend.get_ee_position(), dtype=np.float32)
        width = float(self.backend.get_gripper_width())
        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        ee_z, top_z, bottom_z = float(ee[2]), float(top[2]), float(bottom[2])
        if self.z_canonicalizer is not None:
            # Real-only: shift the vertical channels into the training table frame using the
            # table height measured from the resting cube(s). The lifted (top) cube is tracked
            # continuously — never snapped — so the lift is visible immediately. Deltas are
            # frame-invariant, so the action/target path needs no matching correction.
            delta = self.z_canonicalizer.delta(bottom_z, top_z)
            ee_z += delta
            top_z += delta
            bottom_z += delta
        feats = [float(ee[0]), float(ee[1]), ee_z, width,
                 float(top[0]), float(top[1]), top_z,
                 float(bottom[0]), float(bottom[1])]
        if self.obs_include_bottom_z:
            feats.append(bottom_z)
        return np.array(feats, dtype=np.float32)

    def _pose_staleness(self) -> dict:
        """Per-cube pose age + a combined stale flag for the step info.

        Flag-only: the policy keeps acting on the last good pose (occlusion is
        routine during a stack), but a human MILE intervener can see when the
        observation is being held rather than freshly perceived.
        """
        src = self.pose_source
        ages = {name: src.pose_age(name) for name in (TOP_CUBE, BOTTOM_CUBE)}
        stale = any(src.is_stale(name) for name in (TOP_CUBE, BOTTOM_CUBE))
        return {"pose_stale": bool(stale), "pose_age": ages}

    def _is_success(self) -> bool:
        c = self.config
        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        xy_off = float(np.linalg.norm(top[:2] - bottom[:2]))
        target_z = float(bottom[2]) + c.cube_size
        z_err = abs(float(top[2]) - target_z)
        released = self.backend.get_gripper_width() > c.gripper_open_width - 0.01
        top_pos = top[:3].astype(np.float32)
        pos_delta = (
            np.inf if self._last_success_top_pos is None
            else float(np.linalg.norm(top_pos - self._last_success_top_pos))
        )
        self._last_success_pos_delta = pos_delta
        stable = (
            z_err < c.success_z_tol
            and released
            and (
                self._last_success_top_pos is None
                or pos_delta < c.success_pos_stable_tol
            )
        )
        if stable:
            self._success_stable_count += 1
        else:
            self._success_stable_count = 0
        self._last_success_top_pos = top_pos
        return self._success_stable_count >= c.success_stable_steps

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.backend.reset(self.np_random)
        self._ee_target = np.asarray(
            self.backend.get_ee_position(), dtype=np.float32).copy()
        self._step_count = 0
        self._success_stable_count = 0
        self._last_success_top_pos = None
        self._last_success_pos_delta = float("inf")
        return self._build_obs(), {"success": 0}

    def step(self, action):
        c = self.config
        action = np.asarray(action, dtype=np.float32).reshape(4)
        # Anchor the commanded target to the actual EE position so the accumulator
        # cannot overshoot the waypoint and oscillate when the controller lags.
        actual_ee = np.asarray(self.backend.get_ee_position(), dtype=np.float32)
        self._ee_target = np.clip(
            actual_ee + action[:3] * c.action_scale,
            c.workspace_low, c.workspace_high).astype(np.float32)
        # Command the backend's own wrist-down orientation (the one _home() settles to),
        # not the module default: the fr3 stack needs FR3_DOWN_QUAT, and sending the
        # multipanda DOWN_QUAT instead made the FR3 Cartesian-pose controller hold on a
        # discontinuous orientation so the arm never moved. Backends without a configured
        # down_quat (the fake kinematic world) ignore orientation, so the fallback is inert.
        self.backend.set_equilibrium_pose(
            self._ee_target, getattr(self.backend, "down_quat", DOWN_QUAT))
        self.backend.set_gripper(float(action[3]))

        obs = self._build_obs()
        success = self._is_success()

        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        xy_off = float(np.linalg.norm(top[:2] - bottom[:2]))
        z_err = abs(float(top[2]) - (float(bottom[2]) + c.cube_size))
        reward = -xy_off - z_err + (10.0 if success else 0.0)

        self._step_count += 1
        terminated = bool(success)
        truncated = self._step_count >= c.max_steps
        return obs, float(reward), terminated, truncated, {
            "success": int(success),
            "success_stable_count": self._success_stable_count,
            "xy_off": xy_off,
            "z_err": z_err,
            "top_pos_delta": self._last_success_pos_delta,
            **self._pose_staleness(),
        }

    def close(self):
        self.backend.close()
