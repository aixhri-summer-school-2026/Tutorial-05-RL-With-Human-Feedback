import numpy as np
import pytest

from mile_franka.pose.base import Pose
from mile_franka.viz.mujoco_twin import pose_to_freejoint_qpos


def test_pose_to_freejoint_qpos_reorders_quat_to_wxyz():
    # Pose.orientation is (qx, qy, qz, qw); MuJoCo free joints store (x,y,z, qw,qx,qy,qz).
    pose = Pose(position=[0.1, 0.2, 0.3], orientation=[0.0, 0.0, 0.7071, 0.7071])
    q = pose_to_freejoint_qpos(pose)
    assert q.shape == (7,)
    np.testing.assert_allclose(q[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(q[3:], [0.7071, 0.0, 0.0, 0.7071], atol=1e-6)
