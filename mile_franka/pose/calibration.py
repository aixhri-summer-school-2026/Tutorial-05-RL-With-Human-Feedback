"""Camera↔base calibration file format for the eye-to-hand RealSense.

``camera_to_base`` is a 4×4 homogeneous transform with p_base = T @ p_cam, i.e. the camera
frame expressed in the robot base frame. That is exactly the parent→child transform a
static_transform_publisher needs with parent=panda_link0, child=camera optical frame, so
tf2 can chain base → camera → tag. Real values are machine-specific (regenerate with the
calibration capture script after any camera move); only the template is committed.

The lower half of this module (ChArUcoBoard, detect_charuco_pose, solve_eye_to_hand) is the
pure calibration logic consumed by ``scripts/calibrate_camera.py``. All OpenCV imports are
lazy so the dataclasses, loader, and static-transform emission import without cv2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

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

    @property
    def camera_matrix(self) -> np.ndarray:
        """3×3 intrinsic matrix (for OpenCV)."""
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    def static_transform_args(self, parent_frame: str, child_frame: str) -> List[str]:
        """Args for ros2 static_transform_publisher (x y z qx qy qz qw parent child)."""
        from scipy.spatial.transform import Rotation

        t = self.camera_to_base[:3, 3]
        quat = Rotation.from_matrix(self.camera_to_base[:3, :3]).as_quat()  # xyzw
        vals = [f"{v:g}" for v in (*t, *quat)]
        return [*vals, parent_frame, child_frame]

    def save(self, path: str) -> None:
        """Write this calibration to a YAML file."""
        mat_list = self.camera_to_base.tolist()
        data = {
            "intrinsics": {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy},
            "camera_to_base": mat_list,
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


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


# ---------------------------------------------------------------------------
# Calibration capture primitives (pure — no ROS, testable with synthetic data)
# ---------------------------------------------------------------------------

@dataclass
class ChArUcoBoard:
    """Parameters of a ChArUco calibration target mounted on the gripper."""

    dict_name: str = "DICT_4X4_50"
    squares_x: int = 5
    squares_y: int = 7
    square_length: float = 0.04   # metres (side of a chessboard square)
    marker_length: float = 0.03   # metres (side of an ArUco marker inside the square)

    @property
    def _dictionary(self):
        import cv2
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))

    @property
    def _board(self):
        import cv2
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length, self.marker_length,
            self._dictionary)
        # OpenCV >= 4.7 changed the default marker layout within each ChArUco square
        # ("legacy" vs new pattern). This lab's physical board was generated with the
        # legacy layout — without this, CharucoDetector finds the individual ArUco
        # markers fine but interpolateCornersCharuco silently returns zero corners,
        # since it's looking for markers in the wrong sub-cell position.
        board.setLegacyPattern(True)
        return board


@dataclass
class CalibSample:
    """One capture: the EE pose reported by the controller + the board pose seen by the camera."""
    ee_pose: np.ndarray       # 4×4  O_T_EE (gripper in panda_link0)
    board_pose: np.ndarray    # 4×4  camera→board (board in camera optical frame)


# cv2.solvePnP's default DLT init requires at least 6 3D-2D correspondences.
_MIN_PNP_POINTS = 6


def detect_charuco_pose(board: ChArUcoBoard, image: np.ndarray,
                        camera_matrix: np.ndarray,
                        dist_coeffs: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Detect the ChArUco board in *image* and return camera→board (4×4), or None.

    *image* is a BGR uint8 array (H, W, 3).  *camera_matrix* is the 3×3 intrinsic matrix.
    At least 4 detected ChArUco corners are required for a valid pose.
    """
    import cv2

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # OpenCV >= 4.7 overhauled the aruco API: the free functions detectMarkers /
    # interpolateCornersCharuco / estimatePoseCharucoBoard are gone, replaced by the
    # CharucoDetector class + matchImagePoints/solvePnP for the board pose.
    detector = cv2.aruco.CharucoDetector(board._board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < _MIN_PNP_POINTS:
        return None
    # Board-frame object points <-> detected image points, then PnP for camera->board.
    obj_points, img_points = board._board.matchImagePoints(charuco_corners, charuco_ids)
    # solvePnP's default DLT needs >= 6 point correspondences; a marginal view with fewer
    # would raise mid-detection (and kill the caller's render thread). Require the floor and
    # guard the solve so a bad frame degrades to "no detection" rather than crashing.
    if obj_points is None or len(obj_points) < _MIN_PNP_POINTS:
        return None
    try:
        valid, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    except cv2.error:
        return None
    if not valid:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()
    return T


def solve_eye_to_hand(samples: List[CalibSample]) -> np.ndarray:
    """Solve eye-to-hand extrinsics from (O_T_EE, camera→board) pairs.

    Returns camera_to_base (4×4) — the camera frame expressed in the robot base frame.
    Requires at least 3 samples with distinct poses.

    Uses the Tsai method (CALIB_HAND_EYE_TSAI), which is the standard for eye-to-hand
    calibration with a target rigidly attached to the end-effector.
    """
    import cv2

    if len(samples) < 3:
        raise ValueError(f"need at least 3 samples, got {len(samples)}")

    R_gripper2base = [s.ee_pose[:3, :3].copy() for s in samples]
    t_gripper2base = [s.ee_pose[:3, 3].copy().reshape(3, 1) for s in samples]
    # calibrateHandEye expects target→camera (board→camera), not camera→board.
    board2cam = [np.linalg.inv(s.board_pose) for s in samples]
    R_target2cam = [T[:3, :3].copy() for T in board2cam]
    t_target2cam = [T[:3, 3].copy().reshape(3, 1) for T in board2cam]

    # OpenCV's calibrateHandEye(CALIB_HAND_EYE_TSAI) solves:
    #   base_T_gripper_i * X = camera_T_base * camera_T_board_i
    # and returns X = gripper_T_board (the board-on-gripper transform), NOT
    # camera_to_base directly.  Recover camera_to_base from the fundamental
    # equation for any sample:
    #   camera_to_base = base_T_gripper_i * X * inv(camera_T_board_i)
    R_g2b, t_g2b = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI)
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R_g2b            # gripper→board rotation (name from OpenCV is misleading)
    X[:3, 3] = t_g2b.squeeze()

    # Average camera_to_base over all samples for noise reduction.
    estimates = []
    for s, b2c in zip(samples, board2cam):
        # camera_to_base = O_T_EE * gripper_T_board * board_T_camera
        est = s.ee_pose @ X @ b2c   # b2c = board→camera = inv(camera→board)
        estimates.append(est)

    camera_to_base = np.mean(estimates, axis=0)
    # Re-orthonormalise the rotation (averaging can pull it off SO(3)).
    u, _, vt = np.linalg.svd(camera_to_base[:3, :3])
    camera_to_base[:3, :3] = u @ vt
    return camera_to_base
