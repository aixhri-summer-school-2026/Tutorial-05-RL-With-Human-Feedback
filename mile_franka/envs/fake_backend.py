from __future__ import annotations

import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.backend import RobotBackend
from mile_franka.pose.base import ObjectPoseSource, Pose

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)  # cubes stay axis-aligned in the fake world

TOP_CUBE = "top_cube"
BOTTOM_CUBE = "bottom_cube"


class FakeWorld:
    """Shared mutable state for the fake backend and its pose source."""

    def __init__(self, config: StackTaskConfig):
        self.config = config
        self.ee_pos = np.zeros(3, dtype=np.float32)
        self.gripper_width = float(config.gripper_open_width)
        self.grasping = False
        self.top_cube = np.zeros(3, dtype=np.float32)
        self.bottom_cube = np.zeros(3, dtype=np.float32)

    def reset(self, np_random: np.random.Generator) -> None:
        c = self.config
        lo = c.workspace_low + c.reset_margin
        hi = c.workspace_high - c.reset_margin
        rest_z = c.table_z + c.cube_size / 2.0
        # Draw two xy centers at least reset_min_separation apart.
        while True:
            bottom_xy = np_random.uniform(lo[:2], hi[:2])
            top_xy = np_random.uniform(lo[:2], hi[:2])
            if np.linalg.norm(top_xy - bottom_xy) >= c.reset_min_separation:
                break
        self.bottom_cube = np.array([*bottom_xy, rest_z], dtype=np.float32)
        self.top_cube = np.array([*top_xy, rest_z], dtype=np.float32)
        # Start the EE above the top cube, gripper open, nothing grasped.
        self.ee_pos = np.array([*top_xy, hi[2]], dtype=np.float32)
        self.gripper_width = float(c.gripper_open_width)
        self.grasping = False

    def set_ee(self, position: np.ndarray) -> None:
        self.ee_pos = np.asarray(position, dtype=np.float32).copy()
        if self.grasping:
            self.top_cube = self.ee_pos.copy()  # grasped cube sits at the gripper

    def set_gripper(self, command: float) -> None:
        c = self.config
        if command > 0:  # close
            self.gripper_width = float(c.gripper_closed_width)
            if not self.grasping and np.linalg.norm(self.ee_pos - self.top_cube) < c.cube_size:
                self.grasping = True
                self.top_cube = self.ee_pos.copy()
        else:  # open / release
            self.gripper_width = float(c.gripper_open_width)
            if self.grasping:
                self.grasping = False
                rest_z = c.table_z + c.cube_size / 2.0
                xy_off = np.linalg.norm(self.top_cube[:2] - self.bottom_cube[:2])
                if xy_off < c.success_xy_tol:  # aligned: snap onto the bottom cube
                    self.top_cube = np.array(
                        [*self.bottom_cube[:2], rest_z + c.cube_size], dtype=np.float32)
                else:  # misaligned: drop to the table under the EE
                    self.top_cube = np.array(
                        [self.ee_pos[0], self.ee_pos[1], rest_z], dtype=np.float32)


class FakeRobotBackend(RobotBackend):
    """RobotBackend backed by a FakeWorld (no ROS)."""

    def __init__(self, world: FakeWorld):
        self.world = world

    def reset(self, np_random: np.random.Generator) -> None:
        self.world.reset(np_random)

    def set_equilibrium_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        self.world.set_ee(position)  # orientation ignored: fake wrist is always down

    def set_gripper(self, command: float) -> None:
        self.world.set_gripper(command)

    def get_ee_position(self) -> np.ndarray:
        return self.world.ee_pos.copy()

    def get_gripper_width(self) -> float:
        return float(self.world.gripper_width)


class WorldPoseSource(ObjectPoseSource):
    """ObjectPoseSource reading cube poses from a shared FakeWorld."""

    def __init__(self, world: FakeWorld):
        self.world = world

    def get_pose(self, name: str) -> Pose:
        if name == TOP_CUBE:
            return Pose(tuple(self.world.top_cube), IDENTITY_QUAT)
        if name == BOTTOM_CUBE:
            return Pose(tuple(self.world.bottom_cube), IDENTITY_QUAT)
        raise KeyError(f"unknown object: {name!r}")
