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
                 config: Optional[StackTaskConfig] = None, mode: str = "sim"):
        super().__init__()
        self.backend = backend
        self.pose_source = pose_source
        self.config = config if config is not None else StackTaskConfig()
        self.mode = mode

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        self._ee_target = np.zeros(3, dtype=np.float32)
        self._step_count = 0

    def _build_obs(self) -> np.ndarray:
        ee = np.asarray(self.backend.get_ee_position(), dtype=np.float32)
        width = np.array([self.backend.get_gripper_width()], dtype=np.float32)
        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        return np.concatenate([ee, width, top, bottom]).astype(np.float32)

    def _is_success(self) -> bool:
        c = self.config
        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        xy_off = float(np.linalg.norm(top[:2] - bottom[:2]))
        target_z = float(bottom[2]) + c.cube_size
        z_err = abs(float(top[2]) - target_z)
        released = self.backend.get_gripper_width() > c.gripper_closed_width + 1e-4
        return xy_off < c.success_xy_tol and z_err < c.success_z_tol and released

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.backend.reset(self.np_random)
        self._ee_target = np.asarray(
            self.backend.get_ee_position(), dtype=np.float32).copy()
        self._step_count = 0
        return self._build_obs(), {"success": 0}

    def step(self, action):
        c = self.config
        action = np.asarray(action, dtype=np.float32).reshape(4)
        self._ee_target = np.clip(
            self._ee_target + action[:3] * c.action_scale,
            c.workspace_low, c.workspace_high).astype(np.float32)
        self.backend.set_equilibrium_pose(self._ee_target, DOWN_QUAT)
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
        return obs, float(reward), terminated, truncated, {"success": int(success)}

    def close(self):
        self.backend.close()
