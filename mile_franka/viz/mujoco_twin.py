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


def _franka_description_franka_dir() -> str:
    """Path to franka_description's mujoco/franka dir (where panda.xml + assets live).

    Lazy ament import so the module loads without ROS for the pure tests.
    """
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("franka_description"),
                        "mujoco", "franka")


def resolve_scene_path() -> str:
    """Copy the stacking scene + objects beside franka_description's panda.xml and
    return the scene path.

    The scene uses relative includes (panda.xml, meshdir="assets"), so it must sit
    in that directory at load time. scripts/sim_up.sh does this for the sim; the
    real-robot path never runs sim_up.sh, so the twin repeats the injection here.
    """
    dest = _franka_description_franka_dir()
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "assets", "mujoco")
    for fn in ("stacking_scene.xml", "stacking_objects.xml"):
        shutil.copy(os.path.join(src_dir, fn), os.path.join(dest, fn))
    return os.path.join(dest, "stacking_scene.xml")
