import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mile_franka.pose.apriltag import cube_center_pose


def test_offset_along_z_with_identity_rotation():
    pose = cube_center_pose(translation=[0.0, 0.0, 0.5],
                            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                            half_edge=0.025)
    # AprilTag +z points OUT of the tag; cube center is BEHIND the face (-z).
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.475], atol=1e-6)
    np.testing.assert_allclose(pose.orientation, [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_offset_follows_orientation():
    q = Rotation.from_euler("x", 180, degrees=True).as_quat()
    # 180° around X → tag z-axis points world -z → -half_edge flips the offset to +z.
    pose = cube_center_pose(translation=[0.0, 0.0, 0.5], quat_xyzw=q, half_edge=0.025)
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.525], atol=1e-6)


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
    # half_edge offset is along -z (into the cube), so 0.5 → 0.475.
    np.testing.assert_allclose(pose.position, [0.0, 0.0, 0.475], atol=1e-6)
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


def test_fresh_two_tuple_lookup_is_not_stale():
    # Injected 2-tuple lookups (no age element) are treated as fresh.
    tf = FakeTf({CUBE_TAG_FRAMES[TOP_CUBE]: ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0])})
    src = AprilTagPoseSource(half_edge=0.025, tf_lookup=tf, max_age=0.5)
    src.get_pose(TOP_CUBE)
    assert src.is_stale(TOP_CUBE) is False
    assert src.pose_age(TOP_CUBE) < 0.5


def test_pose_age_and_stale_unknown_before_first_detection():
    src = AprilTagPoseSource(half_edge=0.025, tf_lookup=FakeTf({}), max_age=0.5)
    assert src.pose_age(TOP_CUBE) is None
    assert src.is_stale(TOP_CUBE) is True  # never seen counts as stale


def test_transform_age_backdates_observation_so_pose_reads_stale():
    # A 3-tuple lookup reports the transform's own age; a cached/old tf must read
    # stale immediately even though get_pose() "succeeded" this frame.
    frame = CUBE_TAG_FRAMES[BOTTOM_CUBE]

    def aged_lookup(base_frame, tag_frame):
        # age = 2.0s, well beyond max_age=0.5
        return ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0], 2.0)

    src = AprilTagPoseSource(half_edge=0.025, tf_lookup=aged_lookup, max_age=0.5)
    src.get_pose(BOTTOM_CUBE)
    assert src.is_stale(BOTTOM_CUBE) is True
    assert src.pose_age(BOTTOM_CUBE) >= 2.0
