from __future__ import annotations

import threading
from typing import Dict

import numpy as np

from mile_franka.pose.base import ObjectPoseSource, Pose


class MujocoGtPoseSource(ObjectPoseSource):
    """ObjectPoseSource subscribing to GT PoseStamped topics from the sim."""

    def __init__(self, node, topics: Dict[str, str]):
        """node: an rclpy Node (reuse the backend's). topics: object name -> topic."""
        from geometry_msgs.msg import PoseStamped

        self._lock = threading.Lock()
        self._latest: Dict[str, Pose] = {}
        for name, topic in topics.items():
            node.create_subscription(
                PoseStamped, topic, self._make_cb(name), 10)

    def _make_cb(self, name: str):
        def _cb(msg) -> None:
            p, o = msg.pose.position, msg.pose.orientation
            with self._lock:
                self._latest[name] = Pose((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        return _cb

    def get_pose(self, name: str) -> Pose:
        with self._lock:
            if name not in self._latest:
                raise KeyError(f"no pose received yet for {name!r}")
            return self._latest[name]
