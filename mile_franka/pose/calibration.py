"""Camera↔base calibration file format for the eye-to-hand RealSense.

`camera_to_base` is a 4x4 homogeneous transform with p_base = T @ p_cam, i.e. the camera
frame expressed in the robot base frame. That is exactly the parent→child transform a
static_transform_publisher needs with parent=panda_link0, child=camera optical frame, so
tf2 can chain base → camera → tag. Real values are machine-specific (regenerate with the
calibration capture script after any camera move); only the template is committed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import yaml


@dataclass
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_base: np.ndarray  # (4, 4)

    @property
    def camera_params(self) -> Tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    def static_transform_args(self, parent_frame: str, child_frame: str) -> List[str]:
        """Args for ros2 static_transform_publisher (x y z qx qy qz qw parent child)."""
        from scipy.spatial.transform import Rotation

        t = self.camera_to_base[:3, 3]
        quat = Rotation.from_matrix(self.camera_to_base[:3, :3]).as_quat()  # xyzw
        vals = [f"{v:g}" for v in (*t, *quat)]
        return [*vals, parent_frame, child_frame]


def load_camera_calibration(path: str) -> CameraCalibration:
    with open(path) as f:
        data = yaml.safe_load(f)
    intr = data["intrinsics"]
    mat = np.asarray(data["camera_to_base"], dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"camera_to_base must be 4x4, got {mat.shape}")
    return CameraCalibration(
        fx=float(intr["fx"]), fy=float(intr["fy"]),
        cx=float(intr["cx"]), cy=float(intr["cy"]),
        camera_to_base=mat,
    )
