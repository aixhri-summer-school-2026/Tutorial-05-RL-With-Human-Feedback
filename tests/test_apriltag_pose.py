import numpy as np
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
