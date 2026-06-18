import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mile_franka.pose.calibration import (
    CalibSample,
    CameraCalibration,
    ChArUcoBoard,
    load_camera_calibration,
    solve_eye_to_hand,
)


def _write_yaml(tmp_path, body):
    p = tmp_path / "calib.yaml"
    p.write_text(body)
    return str(p)


# ---- I/O tests (Task 1) ----

def test_load_camera_calibration_reads_intrinsics_and_extrinsics(tmp_path):
    path = _write_yaml(tmp_path, """
intrinsics: {fx: 615.0, fy: 616.0, cx: 320.0, cy: 240.0}
camera_to_base:
  - [1.0, 0.0, 0.0, 0.2]
  - [0.0, 1.0, 0.0, -0.1]
  - [0.0, 0.0, 1.0, 0.5]
  - [0.0, 0.0, 0.0, 1.0]
""")
    calib = load_camera_calibration(path)
    assert calib.camera_params == (615.0, 616.0, 320.0, 240.0)
    assert calib.camera_to_base.shape == (4, 4)
    np.testing.assert_allclose(calib.camera_to_base[:3, 3], [0.2, -0.1, 0.5])


def test_load_rejects_non_4x4_extrinsics(tmp_path):
    path = _write_yaml(tmp_path, """
intrinsics: {fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0}
camera_to_base: [[1.0, 0.0], [0.0, 1.0]]
""")
    with pytest.raises(ValueError):
        load_camera_calibration(path)


def test_static_transform_args_emits_xyz_quat_and_frames(tmp_path):
    path = _write_yaml(tmp_path, """
intrinsics: {fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0}
camera_to_base:
  - [1.0, 0.0, 0.0, 0.2]
  - [0.0, 1.0, 0.0, -0.1]
  - [0.0, 0.0, 1.0, 0.5]
  - [0.0, 0.0, 0.0, 1.0]
""")
    calib = load_camera_calibration(path)
    args = calib.static_transform_args("panda_link0", "camera_color_optical_frame")
    assert args[:3] == ["0.2", "-0.1", "0.5"]
    np.testing.assert_allclose([float(a) for a in args[3:7]], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert args[7:] == ["panda_link0", "camera_color_optical_frame"]


def test_save_and_reload_roundtrip(tmp_path):
    calib = CameraCalibration(fx=615.0, fy=616.0, cx=320.0, cy=240.0,
                              camera_to_base=np.eye(4))
    path = str(tmp_path / "calib.yaml")
    calib.save(path)
    loaded = load_camera_calibration(path)
    assert loaded.camera_params == calib.camera_params
    np.testing.assert_allclose(loaded.camera_to_base, calib.camera_to_base)


def test_camera_matrix_property():
    calib = CameraCalibration(fx=615.0, fy=616.0, cx=320.0, cy=240.0,
                              camera_to_base=np.eye(4))
    K = calib.camera_matrix
    assert K.shape == (3, 3)
    assert K[0, 0] == 615.0
    assert K[1, 1] == 616.0
    assert K[0, 2] == 320.0
    assert K[1, 2] == 240.0
    assert K[2, 2] == 1.0


# ---- ChArUco board ----

def test_charuco_board_constructs_without_cv2():
    """ChArUcoBoard is a plain dataclass — no OpenCV import at construction time."""
    board = ChArUcoBoard(squares_x=5, squares_y=7, square_length=0.04, marker_length=0.03)
    assert board.squares_x == 5
    assert board.square_length == 0.04
    assert board.marker_length == 0.03


def test_charuco_board_dict_access_lazy_loads_cv2():
    board = ChArUcoBoard(dict_name="DICT_4X4_50")
    d = board._dictionary  # first access triggers lazy cv2 import
    assert d is not None


# ---- Hand-eye solve with synthetic data ----

def _make_transform(rpy_deg, xyz):
    """Build a 4×4 homogeneous transform from RPY (degrees) + translation."""
    R = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def _generate_synthetic_samples(camera_to_base, board_to_gripper, ee_poses, noise_std=0.0):
    """Given ground-truth transforms + a list of O_T_EE poses, return CalibSamples."""
    samples = []
    base_to_camera = np.linalg.inv(camera_to_base)
    for ee in ee_poses:
        # Board in camera frame:  cam_T_board = inv(cam_T_base) * base_T_ee * ee_T_board
        board_pose = base_to_camera @ ee @ board_to_gripper
        if noise_std > 0:
            # Perturb the board pose slightly (simulates corner detection noise)
            noise_vec = np.random.randn(6) * noise_std
            noise_R = Rotation.from_rotvec(noise_vec[:3]).as_matrix()
            board_pose[:3, :3] = board_pose[:3, :3] @ noise_R
            board_pose[:3, 3] += noise_vec[3:] * 0.001
        samples.append(CalibSample(ee_pose=ee.copy(), board_pose=board_pose.copy()))
    return samples


def test_solve_eye_to_hand_recovers_identity():
    """Camera at base origin: solve must recover identity."""
    camera_to_base = np.eye(4)
    board_to_gripper = _make_transform([0, 0, 90], [0.05, 0.0, -0.08])

    ee_poses = [
        _make_transform([0, 0, 0], [0.4, 0.0, 0.25]),
        _make_transform([0, 10, 0], [0.5, 0.1, 0.20]),
        _make_transform([0, -10, 5], [0.45, -0.1, 0.30]),
        _make_transform([5, 0, 10], [0.4, 0.15, 0.22]),
        _make_transform([-5, 15, -5], [0.5, -0.05, 0.28]),
    ]

    samples = _generate_synthetic_samples(camera_to_base, board_to_gripper, ee_poses)
    result = solve_eye_to_hand(samples)

    np.testing.assert_allclose(result, camera_to_base, atol=1e-6)


def test_solve_eye_to_hand_recovers_translated_camera():
    """Camera offset from base: solve must recover the offset."""
    camera_to_base = _make_transform([0, 0, 0], [0.5, 0.0, 0.3])
    board_to_gripper = _make_transform([0, 0, 90], [0.05, 0.0, -0.08])

    ee_poses = [
        _make_transform([10, -5, 0], [0.4, 0.0, 0.25]),
        _make_transform([0, 20, 0], [0.5, 0.1, 0.20]),
        _make_transform([-10, 0, -5], [0.45, -0.1, 0.30]),
        _make_transform([5, 10, 15], [0.4, 0.15, 0.22]),
        _make_transform([-5, -10, 5], [0.5, -0.05, 0.28]),
        _make_transform([15, 5, -10], [0.42, 0.08, 0.26]),
    ]

    samples = _generate_synthetic_samples(camera_to_base, board_to_gripper, ee_poses)
    result = solve_eye_to_hand(samples)

    np.testing.assert_allclose(result[:3, 3], camera_to_base[:3, 3], atol=1e-6)
    np.testing.assert_allclose(result[:3, :3], camera_to_base[:3, :3], atol=1e-6)


def test_solve_eye_to_hand_recovers_rotated_camera():
    """Camera rotated 30° about z + offset: solve must recover rotation and translation."""
    camera_to_base = _make_transform([0, 0, 30], [0.6, -0.1, 0.35])
    board_to_gripper = _make_transform([0, 0, 90], [0.05, 0.0, -0.08])

    rng = np.random.RandomState(42)
    ee_poses = []
    for _ in range(10):
        rpy = rng.uniform(-20, 20, 3)
        xyz = rng.uniform([0.35, -0.2, 0.15], [0.7, 0.2, 0.35])
        ee_poses.append(_make_transform(rpy, xyz))

    samples = _generate_synthetic_samples(camera_to_base, board_to_gripper, ee_poses)
    result = solve_eye_to_hand(samples)

    np.testing.assert_allclose(result[:3, 3], camera_to_base[:3, 3], atol=1e-6)
    np.testing.assert_allclose(result[:3, :3], camera_to_base[:3, :3], atol=1e-6)


def test_solve_eye_to_hand_with_noise():
    """Small detection noise should not break the solve (sub-mm / sub-degree recovery)."""
    camera_to_base = _make_transform([0, 5, -20], [0.55, 0.05, 0.32])
    board_to_gripper = _make_transform([0, 0, 90], [0.05, 0.0, -0.08])

    rng = np.random.RandomState(7)
    ee_poses = []
    for _ in range(15):
        rpy = rng.uniform(-20, 20, 3)
        xyz = rng.uniform([0.35, -0.2, 0.15], [0.7, 0.2, 0.35])
        ee_poses.append(_make_transform(rpy, xyz))

    # ~0.5 mm / 0.03° noise per axis — plausible for a 640×480 camera at ~0.7 m
    samples = _generate_synthetic_samples(camera_to_base, board_to_gripper, ee_poses,
                                          noise_std=0.0005)
    result = solve_eye_to_hand(samples)

    # Translation within 2 mm, rotation within 0.5°
    np.testing.assert_allclose(result[:3, 3], camera_to_base[:3, 3], atol=0.002)
    angle_diff = np.arccos(
        (np.trace(result[:3, :3].T @ camera_to_base[:3, :3]) - 1) / 2)
    assert abs(angle_diff) < np.deg2rad(0.5)


def test_solve_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 3"):
        solve_eye_to_hand([])
    with pytest.raises(ValueError, match="at least 3"):
        solve_eye_to_hand([CalibSample(ee_pose=np.eye(4), board_pose=np.eye(4))])
