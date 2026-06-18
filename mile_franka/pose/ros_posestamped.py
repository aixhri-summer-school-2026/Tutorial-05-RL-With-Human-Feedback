"""Generic ObjectPoseSource over geometry_msgs/PoseStamped topics.

The seam for any ROS pose publisher: AprilTag-as-PoseStamped, or hucebot's FoundationPose
node (same message type), drops in by pointing `topics` at it. The reader is injectable so
the mapping logic tests without ROS. Messages are assumed already in the robot base frame
(publish them in panda_link0, or run the publisher with the calibration static transform).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from mile_franka.pose.base import ObjectPoseSource, Pose

# reader(topic) -> (translation[3], quat_xyzw[4]) for the latest msg, or None if none yet.
PoseReader = Callable[[str], Optional[tuple]]


class RosPoseStampedSource(ObjectPoseSource):
    def __init__(self, topics: Dict[str, str], reader: Optional[PoseReader] = None,
                 node=None, base_frame: str = "panda_link0"):
        self.topics = topics
        self.base_frame = base_frame
        self._reader = reader
        self._node = node

    def _read(self) -> PoseReader:
        if self._reader is not None:
            return self._reader
        self._reader = _build_subscriber_reader(self._node, self.topics, self.base_frame)
        return self._reader

    def get_pose(self, name: str) -> Pose:
        topic = self.topics[name]
        latest = self._read()(topic)
        if latest is None:
            raise RuntimeError(f"no PoseStamped received yet on {topic!r} for {name!r}")
        translation, quat = latest
        return Pose(tuple(float(v) for v in translation),
                    tuple(float(v) for v in quat))


def _build_subscriber_reader(node, topics, base_frame) -> PoseReader:
    """Default rclpy reader caching the latest PoseStamped per topic. Lazy ROS import."""
    import rclpy
    from geometry_msgs.msg import PoseStamped

    latest: Dict[str, tuple] = {}

    def _make_cb(topic):
        def _cb(msg: "PoseStamped"):
            p, o = msg.pose.position, msg.pose.orientation
            latest[topic] = ((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        return _cb

    for topic in set(topics.values()):
        node.create_subscription(PoseStamped, topic, _make_cb(topic), 1)

    def reader(topic):
        rclpy.spin_once(node, timeout_sec=0.02)
        return latest.get(topic)

    return reader
