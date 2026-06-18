#!/usr/bin/env python3
"""Interactive eye-to-hand camera calibration for the third-person RealSense.

Captures (image, O_T_EE) pairs while the operator drives the arm to varied poses,
detects a ChArUco board mounted on the gripper in each image, runs
``cv2.calibrateHandEye``, and writes ``config/camera_calib.yaml``.

**Prerequisites (all in-container, all on the same DDS graph):**
- hucebot multipanda_ros2 controller running on the FR3 (or sim)
- ``make apriltag-up`` or equivalent (realsense2_camera publishing color images)
- A ChArUco board rigidly attached to the gripper, facing the camera
- DISPLAY set if you want the live preview window (optional — works headless)

**Usage (inside ``make shell``):**

    python3 scripts/calibrate_camera.py                     # use defaults
    python3 scripts/calibrate_camera.py --samples 15         # collect 15 poses
    python3 scripts/calibrate_camera.py --no-preview         # headless (no imshow)
    python3 scripts/calibrate_camera.py --out /tmp/test.yaml # dry-run output

**Workflow:**
1. The script subscribes to the camera image and the EE pose topic.
2. For each sample: drive the arm to a new pose (joystick or teach pendant), let it
   settle, then press Enter. The script captures one synchronised (image, O_T_EE) pair
   and attempts ChArUco detection.  Valid detections are stored; failures are discarded
   (re-position the arm and try again).
3. After >= *samples* valid captures, press 's' (solve) or collect more then 's'.
4. The script runs the hand-eye solve and writes ``config/camera_calib.yaml``.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# Lazy ROS imports — the script starts faster and the help text works without ROS.
_rclpy = None
_PoseStamped = None

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUT = os.path.join(_REPO, "config", "camera_calib.yaml")
_DEFAULT_INTRINSICS = (615.0, 615.0, 320.0, 240.0)

# Default ChArUco board: 5×7 squares, 40 mm square side, 30 mm marker side.
# Tune these to match the physical board on the gripper.
_DEFAULT_BOARD = dict(squares_x=5, squares_y=7, square_length=0.04,
                      marker_length=0.03, dict_name="DICT_4X4_50")

# Topics — must match the running realsense + controller.
_IMAGE_TOPIC = "/camera/camera/color/image_raw"
_EE_TOPIC = "/cartesian_impedance/cartesian_pos_curr"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lazy_ros():
    global _rclpy, _PoseStamped
    if _rclpy is None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        _rclpy = rclpy
        _PoseStamped = PoseStamped
    return _rclpy, _PoseStamped


def _ee_from_msg(msg) -> np.ndarray:
    """Convert a PoseStamped to a 4×4 O_T_EE homogeneous matrix."""
    p = msg.pose.position
    q = msg.pose.orientation
    # scipy Rotation expects xyzw
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from mile_franka.pose.calibration import (
        CalibSample, CameraCalibration, ChArUcoBoard,
        detect_charuco_pose, solve_eye_to_hand,
    )

    ap = argparse.ArgumentParser(description="Eye-to-hand camera calibration")
    ap.add_argument("--samples", type=int, default=10,
                    help="minimum valid captures before solving (default 10)")
    ap.add_argument("--out", default=_DEFAULT_OUT,
                    help=f"output YAML path (default {_DEFAULT_OUT})")
    ap.add_argument("--no-preview", action="store_true",
                    help="skip the live camera preview window")
    ap.add_argument("--image_topic", default=_IMAGE_TOPIC)
    ap.add_argument("--ee_topic", default=_EE_TOPIC)
    ap.add_argument("--fx", type=float, default=_DEFAULT_INTRINSICS[0])
    ap.add_argument("--fy", type=float, default=_DEFAULT_INTRINSICS[1])
    ap.add_argument("--cx", type=float, default=_DEFAULT_INTRINSICS[2])
    ap.add_argument("--cy", type=float, default=_DEFAULT_INTRINSICS[3])
    ap.add_argument("--square_length", type=float, default=_DEFAULT_BOARD["square_length"])
    ap.add_argument("--marker_length", type=float, default=_DEFAULT_BOARD["marker_length"])
    ap.add_argument("--squares_x", type=int, default=_DEFAULT_BOARD["squares_x"])
    ap.add_argument("--squares_y", type=int, default=_DEFAULT_BOARD["squares_y"])
    ap.add_argument("--dict_name", default=_DEFAULT_BOARD["dict_name"])
    args = ap.parse_args()

    rclpy, PoseStamped = _lazy_ros()
    rclpy.init()
    node = rclpy.create_node("calibrate_camera")

    board = ChArUcoBoard(dict_name=args.dict_name,
                         squares_x=args.squares_x, squares_y=args.squares_y,
                         square_length=args.square_length,
                         marker_length=args.marker_length)
    camera_matrix = np.array([[args.fx, 0, args.cx],
                              [0, args.fy, args.cy],
                              [0, 0, 1]], dtype=np.float64)

    # Shared state between the main thread and the ROS spin thread.
    latest_image: np.ndarray | None = None
    latest_ee: np.ndarray | None = None

    def _image_cb(msg):
        nonlocal latest_image
        latest_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)

    def _ee_cb(msg):
        nonlocal latest_ee
        latest_ee = _ee_from_msg(msg)

    from sensor_msgs.msg import Image as ImageMsg
    node.create_subscription(ImageMsg, args.image_topic, _image_cb, 1)
    node.create_subscription(PoseStamped, args.ee_topic, _ee_cb, 1)

    samples: list[CalibSample] = []

    print(f"Image topic : {args.image_topic}")
    print(f"EE topic    : {args.ee_topic}")
    print(f"Board       : {args.dict_name} {args.squares_x}x{args.squares_y}"
          f" sq={args.square_length}m mk={args.marker_length}m")
    print(f"Intrinsics  : fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")
    print(f"Min samples : {args.samples}")
    print(f"Output      : {args.out}")
    print()
    print("WORKFLOW:")
    print("  1. Drive the arm to a varied pose (joystick / teach pendant).")
    print("  2. Let it settle, then press Enter to capture.")
    print("  3. Green overlay = valid detection & stored.  Red = no ChArUco found — try again.")
    print(f"  4. Repeat until you have ≥{args.samples} captures, then press 's' to solve & save.")
    print("  5. Press 'q' or Ctrl-C to quit without saving.")
    print()

    if not args.no_preview:
        cv2.namedWindow("calibrate_camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("calibrate_camera", 960, 720)

    try:
        while True:
            # Spin ROS to get latest image + EE
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.1)

            img = latest_image
            ee = latest_ee

            # --- Show preview ---
            display = img.copy() if img is not None else np.zeros(
                (480, 640, 3), dtype=np.uint8)

            if img is not None and ee is not None:
                cam2board = detect_charuco_pose(board, img, camera_matrix)
            else:
                cam2board = None

            if cam2board is not None:
                # Draw detected board
                cv2.aruco.drawDetectedMarkers(
                    display, cv2.aruco.Dictionary_get(board._dictionary.bytesList), None)
                cv2.putText(display, f"Board detected | samples={len(samples)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"No board | samples={len(samples)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ee_str = f"EE: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]" if ee is not None else "EE: waiting..."
            cv2.putText(display, ee_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

            if not args.no_preview:
                cv2.imshow("calibrate_camera", display)
                key = cv2.waitKey(50) & 0xFF
            else:
                # Headless: print prompt and wait for input
                print(f"\n[{len(samples)}/{args.samples}] samples.  "
                      f"EE={'ready' if ee is not None else 'waiting...'}  "
                      f"Board={'visible' if cam2board is not None else 'not detected'}")
                print("  [Enter]=capture  [s]=solve&save  [q]=quit")
                choice = input("> ").strip().lower()
                if choice == "":
                    key = ord("\r")
                elif choice == "s":
                    key = ord("s")
                elif choice == "q":
                    key = ord("q")
                else:
                    key = -1

            # --- Handle key ---
            if key == ord("q"):
                print("Quit without saving.")
                break
            elif key == ord("s"):
                if len(samples) < 3:
                    print(f"Need ≥3 samples to solve (have {len(samples)}).")
                    continue
                print(f"\nSolving with {len(samples)} samples...")
                cam2base = solve_eye_to_hand(samples)

                # Use nominal intrinsics (factory); the solve only provides extrinsics.
                calib = CameraCalibration(
                    fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                    camera_to_base=cam2base)
                calib.save(args.out)

                print(f"Saved {args.out}")
                print(f"camera_to_base:\n{cam2base}")
                print(f"static_transform_args: {calib.static_transform_args('panda_link0', 'camera_color_optical_frame')}")
                break
            elif key in (ord("\r"), ord("\n"), 13, 10):
                if img is None or ee is None:
                    print("  Waiting for image + EE data...")
                    continue
                if cam2board is None:
                    print(f"  No ChArUco board detected — reposition the arm and try again.")
                    continue
                samples.append(CalibSample(ee_pose=ee.copy(),
                                           board_pose=cam2board.copy()))
                print(f"  Captured sample {len(samples)}/{args.samples}"
                      f"  EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if not args.no_preview:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
