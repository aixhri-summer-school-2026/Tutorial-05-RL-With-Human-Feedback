"""Object-pose abstraction.

Implementations: MujocoGtPoseSource (sim), AprilTagPoseSource (real, primary),
FoundationPosePoseSource (real, optional markerless). All return a 7-vector
[x, y, z, qx, qy, qz, qw] so the env observation builder is source-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Pose:
    """6-DoF pose as position + quaternion (x, y, z, qx, qy, qz, qw)."""

    position: Sequence[float]
    orientation: Sequence[float]  # quaternion (qx, qy, qz, qw)

    def to_array(self) -> np.ndarray:
        return np.array([*self.position, *self.orientation], dtype=np.float32)


class ObjectPoseSource(ABC):
    """Returns the current pose of a named object in the robot base frame."""

    @abstractmethod
    def get_pose(self, name: str) -> Pose:
        """Return the latest Pose for the object identified by `name`."""
