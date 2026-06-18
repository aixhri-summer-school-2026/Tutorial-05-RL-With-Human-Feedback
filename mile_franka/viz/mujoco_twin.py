"""Pure/isolated helpers for the live MuJoCo digital twin.

Kept free of ROS and the viewer so the non-trivial logic (quaternion order,
joint-name->qpos mapping, scene injection) is unit-testable without hardware.
"""
from __future__ import annotations

import os
import shutil
from typing import List, Sequence, Tuple

import numpy as np

from mile_franka.pose.base import Pose


def pose_to_freejoint_qpos(pose: Pose) -> np.ndarray:
    """Pack a base-frame Pose into MuJoCo free-joint qpos order.

    Pose.orientation is (qx, qy, qz, qw); a MuJoCo free joint stores
    qpos = [x, y, z, qw, qx, qy, qz].
    """
    x, y, z = (float(v) for v in pose.position)
    qx, qy, qz, qw = (float(v) for v in pose.orientation)
    return np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)


def joint_writes(model, names: Sequence[str],
                 positions: Sequence[float]) -> List[Tuple[int, float]]:
    """Map (joint name, position) pairs to (qpos address, value).

    `model` is a mujoco.MjModel. Joint names absent from the model are skipped,
    so a mismatched arm_id or namespaced names degrade gracefully (arm stays at
    rest) instead of erroring. mujoco is imported lazily so this module loads
    without it for the pure tests.
    """
    import mujoco

    writes: List[Tuple[int, float]] = []
    for name, value in zip(names, positions):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        writes.append((int(model.jnt_qposadr[jid]), float(value)))
    return writes
