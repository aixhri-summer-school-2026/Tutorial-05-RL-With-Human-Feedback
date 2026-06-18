import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mile_franka.pose.apriltag import cube_center_pose


def test_offset_along_z_with_identity_rotation():
    pose = cube_center_pose(translation=[0.0, 0.0, 0.5],
                            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                            half_edge=0.025)
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.525], atol=1e-6)
    np.testing.assert_allclose(pose.orientation, [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_offset_follows_orientation():
    q = Rotation.from_euler("x", 180, degrees=True).as_quat()
    pose = cube_center_pose(translation=[0.0, 0.0, 0.5], quat_xyzw=q, half_edge=0.025)
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.475], atol=1e-6)


def test_zero_offset_is_passthrough():
    pose = cube_center_pose(translation=[0.1, 0.2, 0.3],
                            quat_xyzw=[0.0, 0.0, 0.0, 1.0], half_edge=0.0)
    np.testing.assert_allclose(pose.position, [0.1, 0.2, 0.3], atol=1e-6)


from mile_franka.pose.apriltag import AprilTagPoseSource, CUBE_TAG_FRAMES
from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE


class FakeTf:
    """Stand-in for a tf2 lookup: name(frame)->(translation, quat) or raise if unknown."""
    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, base_frame, tag_frame):
        self.calls.append((base_frame, tag_frame))
        if tag_frame not in self.table:
            raise LookupError(tag_frame)
        return self.table[tag_frame]


def test_get_pose_returns_cube_center_for_known_tag():
    tf = FakeTf({CUBE_TAG_FRAMES[BOTTOM_CUBE]: ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0])})
    src = AprilTagPoseSource(half_edge=0.025, base_frame="panda_link0", tf_lookup=tf)
    pose = src.get_pose(BOTTOM_CUBE)
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.525], atol=1e-6)
    assert tf.calls == [("panda_link0", CUBE_TAG_FRAMES[BOTTOM_CUBE])]


def test_get_pose_returns_last_known_when_tag_missing_this_frame():
    table = {CUBE_TAG_FRAMES[TOP_CUBE]: ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0])}
    tf = FakeTf(table)
    src = AprilTagPoseSource(half_edge=0.025, base_frame="panda_link0", tf_lookup=tf)
    first = src.get_pose(TOP_CUBE)
    del table[CUBE_TAG_FRAMES[TOP_CUBE]]
    second = src.get_pose(TOP_CUBE)
    np.testing.assert_allclose(first.position, second.position, atol=1e-9)


def test_get_pose_raises_if_tag_never_seen():
    tf = FakeTf({})
    src = AprilTagPoseSource(half_edge=0.025, base_frame="panda_link0", tf_lookup=tf)
    with pytest.raises(RuntimeError):
        src.get_pose(BOTTOM_CUBE)
