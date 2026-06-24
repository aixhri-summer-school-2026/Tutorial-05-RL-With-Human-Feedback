"""Object-pose abstraction.

Implementations: MujocoGtPoseSource (sim), AprilTagPoseSource (real, primary),
FoundationPosePoseSource (real, optional markerless). All return a 7-vector
[x, y, z, qx, qy, qz, qw] so the env observation builder is source-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

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

    def pose_age(self, name: str) -> Optional[float]:
        """Seconds since the pose for `name` was last actually observed.

        Sources with no notion of staleness (sim ground truth, fake backend)
        always return 0.0 — their pose is exact every step. Real perception
        sources override this to report the true age of the last detection.
        """
        return 0.0

    def is_stale(self, name: str) -> bool:
        """Whether the pose for `name` is older than the source's freshness bound.

        Default False (sources without staleness are never stale). Real sources
        override this against their own max_age.
        """
        return False
