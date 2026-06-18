import numpy as np
import pytest

from mile_franka.pose.calibration import CameraCalibration, load_camera_calibration


def _write_yaml(tmp_path, body):
    p = tmp_path / "calib.yaml"
    p.write_text(body)
    return str(p)


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
