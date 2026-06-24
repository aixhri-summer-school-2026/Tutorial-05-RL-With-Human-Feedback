"""Unit tests for CanonicalCubePoseSource (no ROS/hardware needed)."""
import numpy as np

from mile_franka.pose.base import ObjectPoseSource, Pose
from mile_franka.pose.canonical import (
    CanonicalCubePoseSource, TRAIN_CUBE_QUAT, TablePlaneCanonicalizer)


class _FakeSource(ObjectPoseSource):
    def __init__(self, pose: Pose, age: float = 0.3, stale: bool = True):
        self._pose = pose
        self._age = age
        self._stale = stale

    def get_pose(self, name):
        return self._pose

    def pose_age(self, name):
        return self._age

    def is_stale(self, name):
        return self._stale


def test_orientation_forced_to_training_quat():
    # measured orientation is some arbitrary real AprilTag quaternion
    inner = _FakeSource(Pose((0.5, 0.1, 0.0), (0.5, 0.5, 0.5, -0.5)))
    src = CanonicalCubePoseSource(inner)
    out = src.get_pose("top_cube").to_array()
    assert tuple(out[3:]) == TRAIN_CUBE_QUAT


def test_z_offset_applied_xy_untouched():
    inner = _FakeSource(Pose((0.51, 0.17, -0.0002), (0.5, 0.5, 0.5, -0.5)))
    src = CanonicalCubePoseSource(inner, z_offset=0.025)
    out = src.get_pose("bottom_cube").to_array()
    assert np.isclose(out[0], 0.51)
    assert np.isclose(out[1], 0.17)
    assert np.isclose(out[2], -0.0002 + 0.025)


def test_staleness_delegates_to_inner():
    inner = _FakeSource(Pose((0.0, 0.0, 0.0), (0, 0, 0, 1)), age=0.42, stale=True)
    src = CanonicalCubePoseSource(inner)
    assert src.pose_age("top_cube") == 0.42
    assert src.is_stale("top_cube") is True


# --- TablePlaneCanonicalizer ---

TRAIN_REST_Z = 0.025


def test_resting_cubes_corrected_to_train_rest_z():
    # Real table sits 3 cm high; both cubes rest on it (centers ≈ table + cube_size/2).
    c = TablePlaneCanonicalizer(train_rest_z=TRAIN_REST_Z, snap_tol=0.01)
    bottom_z = 0.03 + TRAIN_REST_Z          # 0.055
    top_z = bottom_z + 0.001                # ~resting, within snap_tol
    delta = c.delta(bottom_z=bottom_z, top_z=top_z)
    # both corrected channels land on the training resting height (denoised average)
    assert np.isclose(bottom_z + delta, TRAIN_REST_Z, atol=1e-3)
    assert np.isclose(top_z + delta, TRAIN_REST_Z, atol=1e-3)


def test_lifted_top_tracked_continuously_not_snapped():
    # Bottom on a 3 cm-high table; top lifted 10 cm above it.
    c = TablePlaneCanonicalizer(train_rest_z=TRAIN_REST_Z, snap_tol=0.01)
    bottom_z = 0.03 + TRAIN_REST_Z
    top_z = bottom_z + 0.10
    delta = c.delta(bottom_z=bottom_z, top_z=top_z)
    # bottom anchors the table -> corrected to train_rest_z exactly (top excluded from estimate)
    assert np.isclose(bottom_z + delta, TRAIN_REST_Z, atol=1e-9)
    # lifted top keeps its real height above the plane -> train_rest_z + lift, no dead-zone
    assert np.isclose(top_z + delta, TRAIN_REST_Z + 0.10, atol=1e-9)


def test_lift_is_monotonic_through_snap_tol_boundary():
    # As the top cube rises from resting through the snap_tol boundary, its corrected z must
    # increase monotonically (no freeze/jump that a naive "snap any near-table cube" would cause).
    c = TablePlaneCanonicalizer(train_rest_z=TRAIN_REST_Z, snap_tol=0.01)
    bottom_z = 0.055
    corrected = [top_z + c.delta(bottom_z, top_z)
                 for top_z in np.linspace(bottom_z, bottom_z + 0.05, 25)]
    assert all(b >= a - 1e-9 for a, b in zip(corrected, corrected[1:])), corrected
