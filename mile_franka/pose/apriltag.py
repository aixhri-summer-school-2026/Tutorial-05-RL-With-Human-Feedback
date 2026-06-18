"""AprilTag object poses from apriltag_ros via tf2.

apriltag_ros publishes a tf frame per detected tag; with a static transform from the robot
base to the camera optical frame (from the calibration YAML), tf2 yields base→tag directly.
The tag sits on a cube FACE, so the cube center is half the cube edge further along the tag's
outward normal (its +z axis). cube_center_pose() is pure so the geometry is unit-tested with
no ROS. CONFIRM@bringup: the offset sign (whether the apriltag +z points into or out of the
cube on the real mount) — flip half_edge sign here if the reported center lands in front of
the face instead of at the cube center.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from mile_franka.pose.base import Pose


def cube_center_pose(translation: Sequence[float], quat_xyzw: Sequence[float],
                     half_edge: float) -> Pose:
    """Cube-center Pose from a base-frame tag transform + half the cube edge.

    The offset is applied along the tag's +z axis expressed in the base frame.
    """
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    R = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    center = t + half_edge * R[:, 2]
    return Pose(tuple(float(v) for v in center),
                tuple(float(v) for v in quat_xyzw))
