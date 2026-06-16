from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RobotBackend(ABC):
    """Cartesian-impedance + gripper control surface, in the robot base frame."""

    @abstractmethod
    def reset(self, np_random: np.random.Generator) -> None:
        """Bring the robot/scene to a start state. Sim randomizes; real is human-gated."""

    @abstractmethod
    def set_equilibrium_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Command the Cartesian equilibrium target: position (3,) + quaternion (4,) xyzw."""

    @abstractmethod
    def set_gripper(self, command: float) -> None:
        """Drive the gripper: command > 0 closes, command <= 0 opens."""

    @abstractmethod
    def get_ee_position(self) -> np.ndarray:
        """Return the current end-effector position (3,) float32."""

    @abstractmethod
    def get_gripper_width(self) -> float:
        """Return the current finger separation in meters."""

    def close(self) -> None:
        """Release any hardware/ROS handles. Default is a no-op."""
