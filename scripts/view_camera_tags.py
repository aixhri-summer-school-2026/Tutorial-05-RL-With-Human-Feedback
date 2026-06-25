#!/usr/bin/env python3
"""Live camera viewer with AprilTag overlay.

Displays the RealSense color stream with detected AprilTag info overlaid.
Two markers are drawn per tag so detection error and pose error are separable:

- **Green** = the raw 2D detection from ``apriltag_ros`` (``/detections``):
  the pixel ``centre`` + ``corners`` the detector actually found in the image.
- **Magenta** = the 3D camera->tag pose (from tf2) reprojected through the
  camera intrinsics. A gap between green and magenta means PnP/pose error;
  a *common* offset of both from the tag means an intrinsics/stream mismatch.

Intrinsics come from the RealSense factory calibration published on the
``camera_info`` topic (not a guessed default and not the hand-eye file).

The preview is served as an MJPEG stream over HTTP — open the URL printed
at startup in any browser.  No X11 / DISPLAY needed.

**Prerequisites (all in-container, same DDS graph):**
- hucebot multipanda_ros2 controller running (sim or real)
- ``make apriltag-up`` or equivalent (realsense2_camera + apriltag_ros)

**Usage (inside ``make shell``):**

    python3 scripts/view_camera_tags.py
    python3 scripts/view_camera_tags.py --no-preview   # text-only, print to stdout
    python3 scripts/view_camera_tags.py --port 8081     # serve on a different port
    python3 scripts/view_camera_tags.py --rate 10        # spin at 10 Hz

**From the host:**

    make view-tags
    # then open http://localhost:8080 in a browser
"""
import argparse
import http.server
import os
import sys
import threading
import time

import cv2
import numpy as np

# Lazy ROS imports — the script starts faster and --help works without ROS.
_rclpy = None

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMAGE_TOPIC = "/camera/camera/color/image_raw"
_CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
_DETECTIONS_TOPIC = "/detections"
_CAMERA_FRAME = "camera_color_optical_frame"
def _default_base_frame() -> str:
    real_stack = os.environ.get("MILE_REAL_STACK", "multipanda").lower()
    if real_stack in ("fr3", "franka_ros2", "fr3_pose"):
        return "fr3_link0"
    return "panda_link0"


_BASE_FRAME = _default_base_frame()
_TAG_FAMILY = "tag36h11"
_TAG_IDS = [0, 1]  # 0 = bottom cube, 1 = top cube
_TAG_NAMES = {0: "bottom", 1: "top"}
# Placeholder intrinsics used ONLY until the first camera_info arrives. The real
# K comes from the RealSense factory calibration on _CAMERA_INFO_TOPIC; the color
# stream runs at 1280x720 (see mile_franka/launch/apriltag_realsense.launch.py).
_DEFAULT_INTRINSICS = (915.0, 915.0, 640.0, 360.0)
_IMAGE_WIDTH, _IMAGE_HEIGHT = 1280, 720

# Inline HTML page with a full-width <img> pulling from /stream.
_HTML_PAGE = """\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Camera + AprilTags</title>
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
    global _rclpy
    if _rclpy is None:
        import rclpy
        _rclpy = rclpy
    return _rclpy


def _default_K():
    """Placeholder camera matrix used until the first camera_info message arrives."""
    fx, fy, cx, cy = _DEFAULT_INTRINSICS
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _project_point(point_3d, K):
    """Project a 3D point (in camera optical frame) to image coordinates.

    Returns (u, v) or None if behind the camera plane.
    """
    if point_3d[2] <= 0.001:
        return None
    u = K[0, 0] * point_3d[0] / point_3d[2] + K[0, 2]
    v = K[1, 1] * point_3d[1] / point_3d[2] + K[1, 2]
    return (int(u), int(v))


def _build_tf2(node):
    """Set up a tf2 Buffer + TransformListener and return (lookup, buffer).

    The lookup function does ``lookup(source, target) -> ((x,y,z), (qx,qy,qz,qw))``
    and raises ``LookupError`` if the transform is unavailable.
    """
    import rclpy as rc
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener

    buffer = Buffer(cache_time=Duration(seconds=10))
    TransformListener(buffer, node)

    # Prime the listener: give DDS discovery enough time to connect to both /tf
    # and /tf_static publishers. The /tf_static latched message (camera→base
    # calibration) needs discovery to complete before the first lookup. 40 spins
    # × 50ms = 2s.
    for _ in range(40):
        rc.spin_once(node, timeout_sec=0.05)

    def lookup(source_frame, target_frame):
        from rclpy.time import Time
        try:
            # timeout=0: the drain loop above already deposited all pending
            # /tf and /tf_static messages into the buffer, so the chain
            # resolves instantly when both transforms are present.
            tf = buffer.lookup_transform(source_frame, target_frame,
                                         Time(), timeout=Duration(seconds=0))
        except Exception as exc:
            raise LookupError(target_frame) from exc
        tr = tf.transform.translation
        rot = tf.transform.rotation
        # Age of the transform: the tf2 buffer caches for ~10s, so a lost tag
        # keeps resolving at "latest" with a stale stamp. Report the age so the
        # caller can drop/dim a pose that hasn't been refreshed by a detection.
        age = (node.get_clock().now() - Time.from_msg(tf.header.stamp)
               ).nanoseconds / 1e9
        return ((tr.x, tr.y, tr.z), (rot.x, rot.y, rot.z, rot.w), age)

    return lookup


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_MAGENTA = (255, 0, 255)        # reprojected 3D pose (fresh)
_MAGENTA_DIM = (128, 0, 128)    # reprojected 3D pose (stale, being held)
_GREEN = (0, 255, 0)            # raw 2D detection (fresh)
_GREEN_DIM = (0, 110, 0)        # raw 2D detection (stale, being held)


def _draw_reprojected_pose(display, uv, tag_id, name, dist, stale=False):
    """Draw the reprojected 3D camera->tag pose (magenta) + label.

    When ``stale`` the pose is dimmed and tagged — it is the last known pose
    being held briefly, not a fresh detection.
    """
    color = _MAGENTA_DIM if stale else _MAGENTA
    cv2.drawMarker(display, uv, color, cv2.MARKER_CROSS, 18, 2)

    label = f"{name} (ID {tag_id})" + (" stale" if stale else "")
    cv2.putText(display, label, (uv[0] + 15, uv[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Distance call-out
    cv2.putText(display, f"{dist:.2f}m", (uv[0] + 15, uv[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _draw_detection(display, det, stale=False):
    """Draw the raw 2D AprilTag detection (green corners + centre).

    ``det`` is an ``apriltag_msgs/AprilTagDetection`` with ``centre`` and four
    ``corners`` in pixel coordinates. When ``stale`` the marker is dimmed.
    """
    color = _GREEN_DIM if stale else _GREEN
    pts = np.array([[int(c.x), int(c.y)] for c in det.corners], dtype=np.int32)
    cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

    centre = (int(det.centre.x), int(det.centre.y))
    cv2.circle(display, centre, 4, color, -1)


def _draw_info_panel(display, lines, k_source):
    """Draw a semi-transparent info panel at the top of the image."""
    if not lines:
        return

    panel_h = 6 + len(lines) * 18 + 6  # padding + text rows + padding
    panel_w = display.shape[1]

    # Semi-transparent dark background
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

    y = 18
    for line in lines:
        cv2.putText(display, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        y += 18

    # Legend + intrinsics source at the bottom.
    cv2.putText(display, "green = 2D detection   magenta = reprojected 3D pose",
                (8, display.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180, 180, 180), 1)
    cv2.putText(display, f"intrinsics: {k_source}", (8, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# MJPEG HTTP server
# ---------------------------------------------------------------------------

_latest_jpeg: bytes = b''
_server_running = True


class _MJpegHandler(http.server.BaseHTTPRequestHandler):
    """Serves a simple HTML page at / and an MJPEG stream at /stream."""

    def log_message(self, format, *args):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/stream":
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_HTML_PAGE.encode())

    def _serve_stream(self):
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


def _start_http_server(port: int) -> threading.Thread:
    server = http.server.HTTPServer(("0.0.0.0", port), _MJpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live camera viewer with AprilTag overlay (MJPEG HTTP)")
    ap.add_argument("--no-preview", action="store_true",
                    help="text-only mode: print tag info to stdout, no HTTP server")
    ap.add_argument("--image-topic", default=_IMAGE_TOPIC,
                    help=f"camera image topic (default: {_IMAGE_TOPIC})")
    ap.add_argument("--rate", type=float, default=15.0,
                    help="target spin rate in Hz (default: 15)")
    ap.add_argument("--port", type=int, default=8080,
                    help="HTTP port for the MJPEG stream (default: 8080)")
    ap.add_argument("--hold", type=float, default=0.5,
                    help="seconds to keep showing a tag after its last detection "
                         "before treating it as lost; dimmed while held (default: 0.5)")
    args = ap.parse_args()

    # A result newer than this is drawn solid; older (but within --hold) is dimmed.
    fresh_window = min(0.15, args.hold)

    rclpy = _lazy_ros()
    rclpy.init()
    node = rclpy.create_node("view_camera_tags")

    # ---- Shared state between the main thread and ROS callbacks ----
    latest_image: np.ndarray | None = None
    # K starts as a placeholder and is replaced by the factory intrinsics the
    # moment the first camera_info arrives; k_source tracks which is in use.
    K = _default_K()
    k_source = "defaults (waiting for camera_info)"
    # Per-tag last detection: tag_id -> (monotonic_time, AprilTagDetection). Held
    # briefly so a dropped frame doesn't make the overlay flicker (see --hold).
    last_det: dict = {}

    def _image_cb(msg):
        nonlocal latest_image
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        # realsense2_camera publishes RGB8; OpenCV expects BGR.
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        latest_image = img

    def _camera_info_cb(msg):
        # RealSense factory intrinsics. msg.k is the 3x3 K matrix, row-major.
        nonlocal K, k_source
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        k_source = f"camera_info ({msg.width}x{msg.height})"

    def _detections_cb(msg):
        # Empty array on a missed frame is normal; only refresh tags we actually
        # saw, so the rest age out via --hold instead of vanishing instantly.
        now = time.monotonic()
        for det in msg.detections:
            last_det[det.id] = (now, det)

    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import Image as ImageMsg
    from apriltag_msgs.msg import AprilTagDetectionArray
    node.create_subscription(ImageMsg, args.image_topic, _image_cb, 1)
    node.create_subscription(CameraInfo, _CAMERA_INFO_TOPIC, _camera_info_cb, 1)
    node.create_subscription(AprilTagDetectionArray, _DETECTIONS_TOPIC,
                             _detections_cb, 1)

    # ---- tf2 for AprilTag lookups (camera→tag only) ----
    tf_lookup = _build_tf2(node)

    tag_frames = {tid: f"{_TAG_FAMILY}:{tid}" for tid in _TAG_IDS}

    # ---- Load camera→base extrinsics for direct composition ----
    # The camera→tag transform comes from tf2 (apriltag_ros publishes it); we
    # compose base_T_tag = camera_to_base @ camera_T_tag in Python instead of
    # relying on the /tf_static chain, which is fragile with CycloneDDS shared
    # memory (latched static transforms can be dropped during discovery).
    _camera_to_base = None  # 4×4 or None
    calib_path = os.environ.get("MILE_CAMERA_CALIB",
                                 os.path.join(_REPO, "config", "camera_calib.yaml"))
    if os.path.isfile(calib_path):
        try:
            from mile_franka.pose.calibration import load_camera_calibration
            _camera_to_base = load_camera_calibration(calib_path).camera_to_base
            print(f"[view-tags] loaded camera→base from {calib_path}")
        except Exception as e:
            print(f"[view-tags] failed to load {calib_path}: {e}")
    else:
        print(f"[view-tags] no {calib_path} — base-frame coords unavailable")

    print(f"Image topic : {args.image_topic}")
    print(f"Rate        : {args.rate} Hz")
    print(f"CameraInfo  : {_CAMERA_INFO_TOPIC}")
    print(f"Detections  : {_DETECTIONS_TOPIC}")
    print(f"Tags        : {list(tag_frames.values())}")
    print(f"Base frame  : {_BASE_FRAME}")
    print(f"Camera frame: {_CAMERA_FRAME}")
    print()

    global _server_running

    if not args.no_preview:
        _start_http_server(args.port)
        print(f"MJPEG stream at http://localhost:{args.port}")
        print(f"Open http://localhost:{args.port} in a browser to view.")
        print("Press Ctrl-C to stop.")
        print()

    period = 1.0 / max(args.rate, 1.0)

    try:
        while True:
            # Drain all pending messages before rendering.  spin_once processes
            # only one message per call; with a single spin the /detections and
            # /tf messages (published by apriltag_ros in the same callback) can
            # land in different slots — the detection callback fires but the tf
            # buffer is empty, so the purple reprojection disappears and flashes.
            # Draining ensures both are consumed before we draw the frame.
            drain_deadline = time.time() + max(period, 0.05)
            while time.time() < drain_deadline:
                rclpy.spin_once(node, timeout_sec=0.005)

            img = latest_image
            if img is None:
                display = np.zeros((_IMAGE_HEIGHT, _IMAGE_WIDTH, 3), dtype=np.uint8)
            else:
                display = img.copy()

            now = time.monotonic()
            tag_lines: list[str] = []

            for tid in _TAG_IDS:
                tag_frame = tag_frames[tid]
                name = _TAG_NAMES[tid]

                # ---- Raw 2D detection (green), held for --hold after last seen ----
                seen = last_det.get(tid)
                det_age = (now - seen[0]) if seen else None
                if seen is not None and det_age <= args.hold:
                    _draw_detection(display, seen[1], stale=det_age > fresh_window)

                # ---- 3D camera->tag pose (magenta), gated on the transform age ----
                # The tf buffer caches ~10s, so check the transform's own age and
                # drop it once it exceeds --hold instead of showing a stale pose.
                try:
                    (cx, cy, cz), cam_quat, tf_age = tf_lookup(_CAMERA_FRAME, tag_frame)
                    if tf_age <= args.hold:
                        cam_pos = (cx, cy, cz)
                        dist = np.sqrt(cx * cx + cy * cy + cz * cz)
                        uv = _project_point(cam_pos, K)
                        if uv is not None:
                            _draw_reprojected_pose(display, uv, tid, name, dist,
                                                   stale=tf_age > fresh_window)

                        # base → tag: compose camera→tag (from tf2) with the
                        # eye-to-hand extrinsics directly, bypassing /tf_static.
                        if _camera_to_base is not None:
                            from scipy.spatial.transform import Rotation as _Rot
                            t_cam = np.array([cx, cy, cz], dtype=np.float64)
                            R_cam = _Rot.from_quat(np.array(cam_quat, dtype=np.float64)).as_matrix()
                            cam_T_tag = np.eye(4, dtype=np.float64)
                            cam_T_tag[:3, :3] = R_cam
                            cam_T_tag[:3, 3] = t_cam
                            base_T_tag = _camera_to_base @ cam_T_tag
                            bx, by, bz = (float(v) for v in base_T_tag[:3, 3])
                            tag_lines.append(
                                f"{name}({tid}): "
                                f"base=[{bx:+.3f},{by:+.3f},{bz:+.3f}]  "
                                f"cam=[{cx:+.3f},{cy:+.3f},{cz:+.3f}]  "
                                f"d={dist:.3f}m"
                                + ("  STALE" if tf_age > fresh_window else ""))
                        else:
                            tag_lines.append(
                                f"{name}({tid}): "
                                f"cam=[{cx:+.3f},{cy:+.3f},{cz:+.3f}]  "
                                f"d={dist:.3f}m  (no calib)")
                    else:
                        tag_lines.append(f"{name}({tid}): lost ({tf_age:.1f}s)")

                except LookupError:
                    tag_lines.append(f"{name}({tid}): not visible")

            # ---- Draw info panel ----
            _draw_info_panel(display, tag_lines, k_source)

            if not args.no_preview:
                # Encode to JPEG for the MJPEG server
                global _latest_jpeg
                _, enc = cv2.imencode(".jpg", display,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                _latest_jpeg = enc.tobytes()
            else:
                # Text-only: print one line, overwriting previous output
                msg = " | ".join(tag_lines) if tag_lines else "waiting for image..."
                print(f"\r{msg}", end="", flush=True)

    except (KeyboardInterrupt, SystemExit):
        print("\nInterrupted.")
    except Exception as exc:
        # rclpy can raise ExternalShutdownException / RCLError during shutdown
        # if spin_once is interrupted mid-wait — treat as a normal exit.
        exc_name = type(exc).__name__
        if exc_name in ("ExternalShutdownException", "RCLError"):
            pass  # normal shutdown race
        elif "rcl_shutdown" in str(exc) or "context is not valid" in str(exc):
            pass
        else:
            raise
    finally:
        _server_running = False
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass  # already shut down is fine


if __name__ == "__main__":
    main()
