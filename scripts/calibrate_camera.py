#!/usr/bin/env python3
"""Interactive eye-to-hand calibration for the third-person camera (RealSense or webcam).

Captures (image, O_T_EE) pairs while the operator drives the arm to varied poses,
detects a ChArUco board mounted on the gripper in each image, runs
``cv2.calibrateHandEye``, and writes ``config/camera_calib.yaml``.

**Prerequisites (all in-container, all on the same DDS graph):**
- hucebot multipanda_ros2 controller running on the FR3 (or sim)
- ``make apriltag-up`` or equivalent (camera driver publishing images; set
  ``MILE_CAMERA=webcam|realsense`` to match, same as ``apriltag-up``)
- A ChArUco board rigidly attached to the gripper, facing the camera

The live preview is served as an **MJPEG stream over HTTP** (open the printed URL
in a browser) — the container ships headless OpenCV, so ``cv2.imshow`` is not
available and no X11 / DISPLAY is needed. Capture commands are entered in the
terminal; the browser just shows the board-detection overlay.

**Usage (inside ``make shell``):**

    python3 scripts/calibrate_camera.py                     # use defaults, stream on :8080
    python3 scripts/calibrate_camera.py --samples 15         # collect 15 poses
    python3 scripts/calibrate_camera.py --no-preview         # text-only (no HTTP stream)
    python3 scripts/calibrate_camera.py --port 8081          # stream on a different port
    python3 scripts/calibrate_camera.py --out /tmp/test.yaml # dry-run output

**Workflow:**
1. The script subscribes to the camera image and the EE pose topic and starts the
   MJPEG preview (open ``http://localhost:<port>`` in a browser).
2. For each sample: drive the arm to a new pose (joystick or teach pendant), let it
   settle, then press Enter in the terminal. The script captures one synchronised
   (image, O_T_EE) pair and attempts ChArUco detection. Valid detections are stored;
   failures are discarded (re-position the arm and try again).
3. After >= *samples* valid captures, press 's' (solve) or collect more then 's'.
4. The script runs the hand-eye solve and writes ``config/camera_calib.yaml``.
"""
import argparse
import http.server
import os
import sys
import threading
import time

import numpy as np

# Suppress OpenCV WARNING spam from drawFrameAxes when the board is near the
# image edge and projected axis endpoints land out of frame (harmless).
# OPENCV_LOG_LEVEL works across all versions; cv2.setLogLevel is 4.10+ only.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

# Lazy ROS imports — the script starts faster and the help text works without ROS.
_rclpy = None
_PoseStamped = None

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUT = os.environ.get("MILE_CAMERA_CALIB",
                              os.path.join(_REPO, "config", "camera_calib.yaml"))
_DEFAULT_INTRINSICS = (615.0, 615.0, 320.0, 240.0)


def _default_base_frame() -> str:
    real_stack = os.environ.get("MILE_REAL_STACK", "multipanda").lower()
    if real_stack in ("fr3", "franka_ros2", "fr3_pose"):
        return "fr3_link0"
    return "panda_link0"

# Default ChArUco board: 11x8 squares, 15 mm square side, 11 mm marker side, DICT_4X4_50 —
# this lab's board on the gripper. Must match CHECKER_SIZE/CHECKER_SQUARE/CHARUCO_MARKER/
# ARUCO_DICT in the Makefile (make calibrate-camera passes them through as CLI overrides;
# these are only the fallback for running this script directly, outside make).
_DEFAULT_BOARD = dict(squares_x=11, squares_y=8, square_length=0.015,
                      marker_length=0.011, dict_name="DICT_4X4_50")

# Topics — must match the running camera driver (realsense or webcam) + controller.
if os.environ.get("MILE_CAMERA", "realsense").lower() == "webcam":
    _IMAGE_TOPIC = "/camera/image_raw"
    _CAMERA_FRAME = "camera_optical_frame"
else:
    _IMAGE_TOPIC = "/camera/camera/color/image_raw"
    _CAMERA_FRAME = "camera_color_optical_frame"
# Primary EE topic: /cartesian_impedance/cartesian_pos_curr (custom_cartesian_impedance_controller,
# multipanda sim). On the real FR3 stack (cartesian_pose_target_controller) this topic exists but
# publishes a stale value; the live EE pose comes from franka_robot_state_broadcaster instead.
_EE_TOPIC = "/cartesian_impedance/cartesian_pos_curr"
_EE_TOPIC_FALLBACK = "/franka_robot_state_broadcaster/current_pose"

# Inline HTML page with a full-width <img> pulling from /stream.
_HTML_PAGE = """\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>calibrate_camera</title>
<style>
  body { margin:0; background:#111; display:flex; flex-direction:column; align-items:center; }
  img  { max-width:100vw; max-height:100vh; }
</style>
</head><body>
<img src="/stream">
</body></html>
"""


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
# Drawing helpers
# ---------------------------------------------------------------------------

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)


def _draw_overlay(img, cam2board, ee, n_samples, min_samples, camera_matrix,
                  joint_angles=None):
    """Render the preview frame: board axes (when detected) + a status panel.

    *img* may be None (no camera yet); returns a BGR uint8 frame ready to encode.
    *joint_angles* is an optional list of 7 arm joint values in radians.
    The board frame is reprojected through *camera_matrix* via ``drawFrameAxes`` —
    a gap between the drawn axes and the physical board signals a pose/intrinsics
    problem, the same diagnostic idea as ``view_camera_tags.py``.
    """
    display = img.copy() if img is not None else np.zeros(
        (480, 640, 3), dtype=np.uint8)

    if cam2board is not None:
        rvec, _ = cv2.Rodrigues(cam2board[:3, :3])
        tvec = cam2board[:3, 3].reshape(3, 1)
        cv2.drawFrameAxes(display, camera_matrix, np.zeros(5),
                          rvec, tvec, 0.04)
        dist = float(np.linalg.norm(cam2board[:3, 3]))
        board_status, board_color = f"Board detected  d={dist:.2f}m", _GREEN
    elif img is not None:
        board_status, board_color = "No board (check position / lighting)", _RED
    else:
        board_status, board_color = "No image yet", _RED

    if img is not None:
        img_status, img_color = f"Image: {img.shape[1]}x{img.shape[0]}", _GREEN
    else:
        img_status, img_color = "Image: waiting...", _RED

    # Build an optional joint-config line (degrees, compact).
    if joint_angles is not None and len(joint_angles) >= 1:
        degs = " ".join(f"{np.degrees(v):+5.0f}" for v in joint_angles[:7])
        joint_line = f"J: [{degs} ]"
    else:
        joint_line = "J: waiting..."

    lines = [
        f"samples: {n_samples}/{min_samples}",
        img_status,
        board_status,
        # ee is the 4x4 O_T_EE matrix; the position is its translation column.
        f"EE: [{ee[0, 3]:+.3f}, {ee[1, 3]:+.3f}, {ee[2, 3]:+.3f}]" if ee is not None
        else "EE: waiting...",
        joint_line,
    ]

    # Semi-transparent panel behind the text.
    panel_h = 6 + len(lines) * 22 + 6
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

    # Colour each status line individually.
    line_colors = [
        _WHITE,        # samples
        img_color,     # image status
        board_color,   # board status
        _WHITE,        # EE
        _WHITE,        # joints
    ]
    y = 24
    for i, line in enumerate(lines):
        lc = line_colors[i] if i < len(line_colors) else _WHITE
        cv2.putText(display, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    lc, 2)
        y += 22

    cv2.putText(display, "terminal: [Enter]=capture  [s]=solve  [q]=quit",
                (8, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1)
    return display


# ---------------------------------------------------------------------------
# MJPEG HTTP server (headless preview — container OpenCV has no GUI/imshow)
# ---------------------------------------------------------------------------

_latest_jpeg: bytes = b''
_server_running = True


class _MJpegHandler(http.server.BaseHTTPRequestHandler):
    """Serves a simple HTML page at / and an MJPEG stream at /stream."""

    def log_message(self, format, *args):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML_PAGE.encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while _server_running:
                    jpeg = _latest_jpeg
                    if jpeg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def _start_http_server(port: int) -> threading.Thread:
    server = http.server.HTTPServer(("0.0.0.0", port), _MJpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


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
                    help="text-only mode: no MJPEG HTTP stream")
    ap.add_argument("--port", type=int, default=8080,
                    help="HTTP port for the MJPEG preview (default 8080)")
    ap.add_argument("--image_topic", default=_IMAGE_TOPIC)
    ap.add_argument("--ee_topic", default=_EE_TOPIC,
                    help=f"primary EE PoseStamped topic (default {_EE_TOPIC}; "
                         f"fallback {_EE_TOPIC_FALLBACK} always subscribed)")
    ap.add_argument("--joint-states-topic", default="/joint_states",
                    help="JointState topic for arm joint display (default /joint_states)")
    ap.add_argument("--base_frame", default=_default_base_frame())
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

    # Shared state between the main thread and the ROS render thread.
    latest_image: np.ndarray | None = None
    latest_ee: np.ndarray | None = None
    latest_ee_stamp: float = 0.0    # time.time() when the last EE message was received
    latest_ee_source: str = ""      # which topic delivered latest_ee (for diagnostics)
    latest_image_stamp: float = 0.0
    latest_js_positions: list[float] | None = None
    # The most recent consistent (image, ee, cam2board) snapshot, updated by the
    # render thread and read by the main thread on capture. Guarded by _lock.
    snapshot: tuple = (None, None, None)
    _lock = threading.Lock()

    def _image_cb(msg):
        nonlocal latest_image, latest_image_stamp
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        # realsense2_camera publishes RGB8; OpenCV / detection expect BGR.
        if getattr(msg, "encoding", "") == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        latest_image = img
        latest_image_stamp = time.time()

    def _ee_cb(msg, source: str = _EE_TOPIC):
        nonlocal latest_ee, latest_ee_stamp, latest_ee_source
        latest_ee = _ee_from_msg(msg)
        latest_ee_stamp = time.time()
        latest_ee_source = source

    def _ee_primary_cb(msg):
        _ee_cb(msg, _EE_TOPIC)

    def _ee_fallback_cb(msg):
        _ee_cb(msg, _EE_TOPIC_FALLBACK)

    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import Image as ImageMsg
    from sensor_msgs.msg import JointState
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    node.create_subscription(ImageMsg, args.image_topic, _image_cb, 1)
    node.create_subscription(PoseStamped, args.ee_topic, _ee_primary_cb, 1)
    # franka_robot_state_broadcaster publishes with BEST_EFFORT reliability; default
    # RELIABLE subscriber will see "incompatible QoS" and receive nothing.
    fallback_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(PoseStamped, _EE_TOPIC_FALLBACK, _ee_fallback_cb, fallback_qos)
    print(f"EE primary  : {args.ee_topic}")
    print(f"EE fallback : {_EE_TOPIC_FALLBACK} (BEST_EFFORT)")

    _js_cb_count = 0
    _js_diag_done = False
    def _js_cb(msg):
        nonlocal latest_js_positions, _js_diag_done, _js_cb_count
        try:
            # Collect arm joints: skip anything with "finger" or "gripper" in the name.
            # The remaining positional joints (panda_joint1..7, fr3_joint1..7, etc.) are
            # sorted by name and displayed.  Works with 1–7 arm joints.
            arm = []
            for name, pos in zip(msg.name, msg.position):
                low = name.lower()
                if "finger" in low or "gripper" in low:
                    continue
                arm.append((name, pos))
            arm.sort(key=lambda kv: kv[0])
            if arm:
                latest_js_positions = [v for _, v in arm[:7]]
                _js_cb_count += 1
                if not _js_diag_done:
                    print(f"[joint_states] arm joint names: {[n for n, _ in arm[:7]]}"
                          f"  topic={_js_topic}", flush=True)
                    _js_diag_done = True
            else:
                if not _js_diag_done:
                    print(f"[joint_states] WARNING: no arm joints found in "
                          f"{list(msg.name)} on {_js_topic}", flush=True)
                    _js_diag_done = True
        except Exception as exc:
            if not _js_diag_done:
                print(f"[joint_states] ERROR in _js_cb: {exc}", flush=True)
                _js_diag_done = True

    # Default joint-states topic depends on the stack.
    _js_topic = getattr(args, "joint_states_topic", "/joint_states")
    node.create_subscription(JointState, _js_topic, _js_cb, 1)

    # Use the camera's own published intrinsics (camera_info), not the 640x480 placeholder
    # defaults: the colour stream runs at 1920x1080, so the wrong K would corrupt every
    # board pose and hence the hand-eye solve. msg.k is the 3x3 K, row-major. The CLI
    # --fx/--fy/--cx/--cy values are a fallback if camera_info never arrives, OR if it
    # arrives but is degenerate — an uncalibrated usb_cam publishes fx=fy=0 until you run
    # 'ros2 run camera_calibration cameracalibrator' for this webcam (see
    # config/webcam_intrinsics.example.yaml). A zero/near-zero focal length would make
    # every PnP solve garbage, so treat it the same as "never arrived".
    info_topic = args.image_topic.rsplit("/", 1)[0] + "/camera_info"
    _got_info = {"k": None}

    def _info_cb(msg):
        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if k[0, 0] > 1.0 and k[1, 1] > 1.0:
            _got_info["k"] = k

    node.create_subscription(CameraInfo, info_topic, _info_cb, 1)
    _t0 = time.time()
    while _got_info["k"] is None and time.time() - _t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if _got_info["k"] is not None:
        camera_matrix = _got_info["k"]
        print(f"Intrinsics  : from camera_info {info_topic} "
              f"(fx={camera_matrix[0,0]:.1f} fy={camera_matrix[1,1]:.1f} "
              f"cx={camera_matrix[0,2]:.1f} cy={camera_matrix[1,2]:.1f})")
    else:
        print(f"Intrinsics  : camera_info not received on {info_topic}; "
              f"using --fx/--fy/--cx/--cy fallback (may be wrong for this resolution)")

    samples: list[CalibSample] = []
    n_samples = 0  # mirror of len(samples) the render thread can read lock-free

    print(f"Image topic : {args.image_topic}")
    print(f"Joint topic : {getattr(args, 'joint_states_topic', '/joint_states')}")
    print(f"Board       : {args.dict_name} {args.squares_x}x{args.squares_y}"
          f" sq={args.square_length}m mk={args.marker_length}m")
    print(f"Intrinsics  : fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")
    print(f"Min samples : {args.samples}")
    print(f"Output      : {args.out}")
    print()

    # ---- Render thread: spin ROS, detect the board, publish preview frames ----
    stop = threading.Event()

    def _render_loop():
        global _latest_jpeg
        while not stop.is_set():
            # Spin enough times to give every subscription a fair shot — each
            # spin_once call processes ONE callback. The timeout must be >0 or
            # CycloneDDS never receives new data (the network recv path needs
            # the wait interval). 15 spins × 10ms = 150ms ceiling, but callbacks
            # return immediately when data is queued, so the typical loop body
            # is much faster.
            for _ in range(15):
                rclpy.spin_once(node, timeout_sec=0.01)
            img = latest_image
            ee = latest_ee
            cam2board = (detect_charuco_pose(board, img, camera_matrix)
                         if img is not None else None)
            with _lock:
                snapshot_local = (None if img is None else img.copy(),
                                  None if ee is None else ee.copy(),
                                  cam2board)
                nonlocal snapshot
                snapshot = snapshot_local
            if not args.no_preview:
                display = _draw_overlay(img, cam2board, ee, n_samples,
                                        args.samples, camera_matrix,
                                        joint_angles=latest_js_positions)
                ok, enc = cv2.imencode(".jpg", display,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    _latest_jpeg = enc.tobytes()

    render_thread = threading.Thread(target=_render_loop, daemon=True)
    render_thread.start()

    global _server_running
    if not args.no_preview:
        _start_http_server(args.port)
        print(f"MJPEG preview at http://localhost:{args.port}")
        print("Open it in a browser to watch board detection (green axes = detected).")
        print()

    print("WORKFLOW:")
    print("  1. Drive the arm to a varied pose (joystick / teach pendant).")
    print("  2. Let it settle, then press Enter to capture.")
    print("  3. Green axes / 'Board detected' = valid; red 'No board' = reposition & retry.")
    print(f"  4. Repeat until you have ≥{args.samples} captures, then press 's' to solve & save.")
    print("  5. Press 'q' or Ctrl-C to quit without saving.")
    print()

    try:
        while True:
            img, ee, cam2board = snapshot
            img_ok = img is not None
            ee_ok = ee is not None
            ready = n_samples >= args.samples
            print(f"[{n_samples}/{args.samples}] "
                  f"Image={'ok' if img_ok else 'waiting...'}  "
                  f"EE={'ready' if ee_ok else 'waiting...'}  "
                  f"Board={'visible' if cam2board is not None else 'not detected'}"
                  f"{'  ← READY TO SOLVE' if ready else ''}")
            prompt = ("  [Enter]=capture  [s]=SOLVE & SAVE  [q]=quit > "
                      if ready
                      else "  [Enter]=capture  [s]=solve&save  [q]=quit > ")
            choice = input(prompt).strip().lower()

            if choice == "q":
                print("Quit without saving.")
                break

            elif choice == "s":
                if len(samples) < 3:
                    print(f"  Need ≥3 samples to solve (have {len(samples)}).")
                    continue
                print(f"\nSolving with {len(samples)} samples...", flush=True)
                try:
                    cam2base = solve_eye_to_hand(samples)
                except Exception as exc:
                    print(f"  Solve failed: {exc}")
                    print("  Try collecting more samples with varied poses and orientations.")
                    continue
                # Store the intrinsics actually used for the pose solve (camera_info when
                # available, else the CLI fallback); the solve itself only provides extrinsics.
                calib = CameraCalibration(
                    fx=float(camera_matrix[0, 0]), fy=float(camera_matrix[1, 1]),
                    cx=float(camera_matrix[0, 2]), cy=float(camera_matrix[1, 2]),
                    camera_to_base=cam2base)
                try:
                    calib.save(args.out)
                except OSError as exc:
                    print(f"  Failed to write {args.out}: {exc}")
                    continue
                print(f"Saved {args.out}", flush=True)
                print(f"camera_to_base:\n{cam2base}", flush=True)
                print("static_transform_args: "
                      f"{calib.static_transform_args(args.base_frame, _CAMERA_FRAME)}",
                      flush=True)
                break

            elif choice == "":
                # ---- Force a fresh EE + image read at capture time ----
                # Spin ROS for up to 2s to ensure the cached EE and image are newer
                # than the moment the user pressed Enter (the render thread's snapshot
                # could be up to 150ms stale, and the controller may have just resumed
                # from Dashboard free-motion / an inactive period).
                ee_before = latest_ee.copy() if latest_ee is not None else None
                img_before = latest_image.copy() if latest_image is not None else None
                ee_stamp_before = latest_ee_stamp
                img_stamp_before = latest_image_stamp
                t_deadline = time.time() + 2.0
                while time.time() < t_deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                    if latest_ee_stamp > ee_stamp_before and latest_image_stamp > img_stamp_before:
                        break
                else:
                    if latest_ee_stamp <= ee_stamp_before:
                        print("  WARNING: EE pose has NOT updated since last read "
                              f"({time.time() - latest_ee_stamp:.1f}s stale) — "
                              "is the controller active?")
                    if latest_image_stamp <= img_stamp_before:
                        print("  WARNING: camera image has NOT updated — "
                              "is the RealSense streaming?")

                # Use the freshest values available (post-spin).
                img = latest_image
                ee = latest_ee

                # Re-detect the board on the freshest image.
                cam2board = (detect_charuco_pose(board, img, camera_matrix)
                             if img is not None else None)

                if img is None or ee is None:
                    print("  Waiting for image + EE data...")
                    continue
                if cam2board is None:
                    print("  No ChArUco board detected — reposition the arm and try again.")
                    continue

                # ---- Sanity check: reject poses that are near-duplicates ----
                MIN_TRANSLATION_DELTA = 0.03   # metres — at least 3 cm from any previous sample
                MIN_ROTATION_DELTA_DEG = 5.0   # degrees — at least 5° from any previous sample
                ee_pos = ee[:3, 3]
                duplicate = False
                for i, prev in enumerate(samples):
                    prev_pos = prev.ee_pose[:3, 3]
                    d_pos = float(np.linalg.norm(ee_pos - prev_pos))
                    # Rotation delta via the geodesic angle between rotation matrices.
                    d_rot = float(
                        np.arccos(min(1.0, (np.trace(prev.ee_pose[:3, :3].T @ ee[:3, :3]) - 1) / 2)))
                    d_rot_deg = np.degrees(d_rot)
                    if d_pos < MIN_TRANSLATION_DELTA and d_rot_deg < MIN_ROTATION_DELTA_DEG:
                        print(f"  REJECTED: pose too similar to sample {i+1} "
                              f"(Δpos={d_pos*100:.0f}cm < {MIN_TRANSLATION_DELTA*100:.0f}cm, "
                              f"Δrot={d_rot_deg:.1f}° < {MIN_ROTATION_DELTA_DEG:.0f}°)."
                              f"  Move the arm farther and try again.")
                        duplicate = True
                        break
                if duplicate:
                    continue

                samples.append(CalibSample(ee_pose=ee.copy(),
                                           board_pose=cam2board.copy()))
                n_samples = len(samples)
                ee_src = latest_ee_source or "unknown"
                print(f"  Captured sample {n_samples}/{args.samples}"
                      f"  EE=[{ee[0, 3]:+.3f}, {ee[1, 3]:+.3f}, {ee[2, 3]:+.3f}]"
                      f"  via {ee_src}")
            else:
                print("  Unrecognised input.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop.set()
        _server_running = False
        render_thread.join(timeout=2.0)
        try:
            node.destroy_node()
        except Exception:
            pass
        # rclpy.shutdown() can hang if spin_once is still in-flight (ROS 2 race);
        # run it in a daemon thread with a timeout so the script always exits cleanly.
        import threading as _thr
        shut_done = _thr.Event()
        def _do_shutdown():
            try:
                rclpy.shutdown()
            except Exception:
                pass
            finally:
                shut_done.set()
        _thr.Thread(target=_do_shutdown, daemon=True).start()
        shut_done.wait(timeout=3.0)  # generous: normal shutdown is <1s


if __name__ == "__main__":
    main()
