import numpy as np
import pytest

from mile_franka.pose.ros_posestamped import RosPoseStampedSource


def test_get_pose_returns_latest_for_name():
    poses = {"object_pose": ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])}
    src = RosPoseStampedSource(topics={"widget": "object_pose"},
                               reader=lambda topic: poses[topic])
    pose = src.get_pose("widget")
    np.testing.assert_allclose(pose.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose.orientation, [0.0, 0.0, 0.0, 1.0])


def test_get_pose_raises_until_message_arrives():
    src = RosPoseStampedSource(topics={"widget": "object_pose"},
                               reader=lambda topic: None)
    with pytest.raises(RuntimeError):
        src.get_pose("widget")
