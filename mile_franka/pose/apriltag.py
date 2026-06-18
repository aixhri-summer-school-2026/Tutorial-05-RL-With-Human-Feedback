"""AprilTag object poses from apriltag_ros via tf2.

apriltag_ros publishes a tf frame per detected tag; with a static transform from the robot
base to the camera optical frame (from the calibration YAML), tf2 yields base→tag directly.
The tag sits on a cube FACE, so the cube center is half the cube edge away from the face
along its normal. cube_center_pose() applies that offset along the tag's +z axis and is pure
so the geometry is unit-tested with no ROS. CONFIRM@bringup: the offset SIGN — whether the
apriltag +z points into the cube (inward, the assumed direction) or out toward the camera
depends on the detector convention and the physical mount; flip the half_edge sign here if
the reported center lands in front of the face instead of at the cube center.
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


import time
from typing import Callable, Dict, Optional, Tuple

from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE
from mile_franka.pose.base import ObjectPoseSource

# tag36h11 ids from scripts/generate_cube_tags.py: 0 = bottom, 1 = top. apriltag_ros names
# each tag's tf frame "<family>:<id>" (configurable in config/apriltag.yaml).
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.036  # printed tag size reported by generate_cube_tags.py
CUBE_TAG_IDS = {BOTTOM_CUBE: 0, TOP_CUBE: 1}
CUBE_TAG_FRAMES = {name: f"{TAG_FAMILY}:{i}" for name, i in CUBE_TAG_IDS.items()}

# A tf lookup: (base_frame, tag_frame) -> (translation[3], quat_xyzw[4]); raises if no tf yet.
TfLookup = Callable[[str, str], Tuple[Tuple[float, ...], Tuple[float, ...]]]


class AprilTagPoseSource(ObjectPoseSource):
    """Cube poses (base frame) from apriltag_ros tf, via an injectable tf lookup."""

    def __init__(self, half_edge: float, base_frame: str = "panda_link0",
                 tf_lookup: Optional[TfLookup] = None, node=None):
        """half_edge: half the cube edge (m). tf_lookup injectable for tests; if None a tf2
        buffer/listener is built on `node` (a live rclpy node) lazily."""
        self.half_edge = float(half_edge)
        self.base_frame = base_frame
        self._tf_lookup = tf_lookup
        self._node = node
        self._last: Dict[str, Tuple[Pose, float]] = {}

    def _lookup(self) -> TfLookup:
        if self._tf_lookup is not None:
            return self._tf_lookup
        self._tf_lookup = _build_tf2_lookup(self._node)
        return self._tf_lookup

    def get_pose(self, name: str) -> Pose:
        frame = CUBE_TAG_FRAMES[name]
        try:
            translation, quat = self._lookup()(self.base_frame, frame)
            pose = cube_center_pose(translation, quat, self.half_edge)
            self._last[name] = (pose, time.time())
        except LookupError:
            pass  # tag not visible this frame; fall through to cached
        if name not in self._last:
            raise RuntimeError(f"AprilTag for {name!r} ({frame}) never seen; "
                               "check the camera/apriltag node and tag visibility")
        return self._last[name][0]

    def last_seen(self, name: str) -> Optional[float]:
        entry = self._last.get(name)
        return None if entry is None else entry[1]


def _build_tf2_lookup(node) -> TfLookup:
    """Default tf2-backed lookup. Imported lazily so the module loads without ROS."""
    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener

    buffer = Buffer(cache_time=Duration(seconds=10))
    TransformListener(buffer, node)

    # Prime the listener: spin enough for subscriptions to connect and for the latched
    # /tf_static publisher to deliver camera->base. Without this warm-up the first
    # lookup_transform(…, Time()) call can fail even when the data is on the wire.
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)

    def lookup(base_frame: str, tag_frame: str):
        rclpy.spin_once(node, timeout_sec=0.02)
        from rclpy.time import Time
        try:
            tf = buffer.lookup_transform(base_frame, tag_frame, Time())
        except Exception as exc:  # tf2 raises LookupException/ExtrapolationException
            raise LookupError(tag_frame) from exc
        tr = tf.transform.translation
        rot = tf.transform.rotation
        return ((tr.x, tr.y, tr.z), (rot.x, rot.y, rot.z, rot.w))

    return lookup
