"""Canonicalize a real pose source to the policy's training observation convention.

The sim ground-truth pose source (MujocoGtPoseSource) reports every cube with a
*constant* orientation quaternion ``[0, 0, -1, 0]`` (xyzw) — the world→base flip of
the identity — and a cube-center z fixed at ``table_z + cube_size/2``.  Because those
obs dims never varied in the BC dataset, the policy's RunningNorm learned a near-zero
std on them (cube quaternion + bottom-cube z).

A real perception source (AprilTagPoseSource) instead reports the cube's *measured*
orientation in the base frame (a completely different quaternion) and a measured z.
Normalizing those with the sim RunningNorm — ``(x - mean) / std`` with std ≈ 1e-6 —
amplifies the deviation into normalized features of magnitude 1e2–1e5, saturating the
network and producing garbage actions on both ``eval-real`` and ``mile-real``.

This wrapper makes the real observation match the training convention: it overwrites
the cube orientation with the canonical quaternion the policy was trained on, and adds
a fixed z offset so the resting cube-center z matches the sim value.  Orientation was
uninformative in training (constant), so discarding the measured orientation loses
nothing the policy could use; the x/y position channels (which carry real variance and
read in-distribution) pass through untouched.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from mile_franka.pose.base import ObjectPoseSource, Pose

# The constant cube quaternion in the BC training observations (xyzw): the world→base
# flip of identity produced by MujocoGtPoseSource (_world_quat_to_base([0,0,0,1])).
TRAIN_CUBE_QUAT = (0.0, 0.0, -1.0, 0.0)


class CanonicalCubePoseSource(ObjectPoseSource):
    """Wrap a pose source, forcing cube orientation + z to the training convention."""

    def __init__(self, inner: ObjectPoseSource,
                 quat_xyzw: Sequence[float] = TRAIN_CUBE_QUAT,
                 z_offset: float = 0.0):
        """inner: the wrapped real source (e.g. AprilTagPoseSource).
        quat_xyzw: orientation to stamp on every cube pose (training convention).
        z_offset: meters added to the measured cube-center z so the resting z matches
            the sim convention (table_z + cube_size/2); the real source typically
            reports rest z ≈ 0 while training used ≈ cube_size/2."""
        self._inner = inner
        self._quat = tuple(float(v) for v in quat_xyzw)
        self._z_offset = float(z_offset)

    def get_pose(self, name: str) -> Pose:
        pose = self._inner.get_pose(name)
        x, y, z = (float(v) for v in pose.position)
        return Pose((x, y, z + self._z_offset), self._quat)

    def pose_age(self, name: str) -> Optional[float]:
        return self._inner.pose_age(name)

    def is_stale(self, name: str) -> bool:
        return self._inner.is_stale(name)


class TablePlaneCanonicalizer:
    """Map the vertical observation channels into the policy's training table frame.

    The table plane is observed directly: the bottom cube always rests on it.  ``delta`` returns
    the offset that shifts a resting cube-center z onto ``train_rest_z`` (the value the policy
    trained on, ``cube_size/2`` with the sim ``table_z = 0``).  The env adds that offset to
    ``ee_z`` and ``top_z`` continuously, so an unknown real table height is corrected without
    retraining.

    The top cube is folded into the table estimate ONLY while it is resting (within ``snap_tol``
    of the bottom cube); once it lifts it is excluded, so a lifted cube never drags the estimate
    and its tracked height stays continuous (it is never snapped).
    """

    def __init__(self, train_rest_z: float, snap_tol: float = 0.01):
        self.train_rest_z = float(train_rest_z)
        self.snap_tol = float(snap_tol)

    def delta(self, bottom_z: float, top_z: float) -> float:
        bottom_z, top_z = float(bottom_z), float(top_z)
        if abs(top_z - bottom_z) < self.snap_tol:
            table_est = 0.5 * (bottom_z + top_z)   # both resting: average to denoise
        else:
            table_est = bottom_z                   # top lifted: anchor on the bottom cube
        return self.train_rest_z - table_est
