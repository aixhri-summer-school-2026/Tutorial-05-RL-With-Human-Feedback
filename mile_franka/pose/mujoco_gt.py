"""Ground-truth object poses from the multipanda MuJoCo sim.

Verified mechanism (2026-06-16): the sim exposes body poses through the mujoco_ros
`get_body_state` SERVICE, not per-object topics. So this source delegates to the backend
(which owns the rclpy node + service client) rather than subscribing. Same single node is
reused, keeping all sim I/O on one synchronously-spun node.
"""
from __future__ import annotations

from mile_franka.pose.base import ObjectPoseSource, Pose


class MujocoGtPoseSource(ObjectPoseSource):
    """ObjectPoseSource reading body poses via the backend's get_body_state service."""

    def __init__(self, backend):
        """backend: a MultipandaRosBackend (exposes get_body_pose(name) -> [x,y,z,qx..qw])."""
        self._backend = backend

    def get_pose(self, name: str) -> Pose:
        a = self._backend.get_body_pose(name)
        return Pose((float(a[0]), float(a[1]), float(a[2])),
                    (float(a[3]), float(a[4]), float(a[5]), float(a[6])))
