# launch/apriltag_webcam.launch.py
"""Bring up a USB/UVC webcam (e.g. Logitech), apriltag_ros, and the camera->base static tf.

Drop-in alternative to apriltag_realsense.launch.py for cameras without a RealSense SDK.

Two *separate* calibration files are involved — they use incompatible YAML schemas and
must not be pointed at the same path:
- Intrinsics (fx/fy/cx/cy + distortion): produced by ``make calibrate-intrinsics``
  (ROS camera_info schema: camera_matrix, distortion_coefficients, rectification_matrix,
  projection_matrix, ...). Path via MILE_WEBCAM_INTRINSICS or config/webcam_intrinsics.yaml.
  Fed to usb_cam's camera_info_url. RealSense doesn't need this — it has factory
  intrinsics on-device.
- Eye-to-hand extrinsics (camera_to_base): produced by scripts/calibrate_camera.py (our
  own schema: intrinsics: {fx,fy,cx,cy}, camera_to_base: 4x4). Path via MILE_CAMERA_CALIB
  or config/camera_calib.yaml. Used for the camera->base static transform below.

Args:
    device:      V4L2 device node (default /dev/video0).
    width:       capture width in pixels (default 1920).
    height:      capture height in pixels (default 1080).
    framerate:   frames per second (default 15).
    camera_info: URL for the intrinsics YAML (default: MILE_WEBCAM_INTRINSICS or
                 config/webcam_intrinsics.yaml).  Must be a file:// URL or package:// URL.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mile_franka.pose.calibration import load_camera_calibration

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIB = (os.environ.get("MILE_CAMERA_CALIB")
         or os.path.join(REPO, "config", "camera_calib.yaml"))
INTRINSICS = (os.environ.get("MILE_WEBCAM_INTRINSICS")
              or os.path.join(REPO, "config", "webcam_intrinsics.yaml"))
APRILTAG_CFG = os.path.join(REPO, "config", "apriltag.yaml")

# usb_cam publishes on <namespace>/image_raw and <namespace>/camera_info (topics are
# absolute, unaffected by node *name* remaps — only the ROS namespace nests them).
# The node below is namespaced "camera", so topics are /camera/image_raw and /camera/camera_info.
CAMERA_IMAGE_TOPIC = "/camera/image_raw"
CAMERA_INFO_TOPIC  = "/camera/camera_info"
# Passed to usb_cam's "frame_id" param below, so images are stamped with this frame.
CAMERA_FRAME = "camera_optical_frame"


def _base_frame() -> str:
    real_stack = os.environ.get("MILE_REAL_STACK", "multipanda").lower()
    if real_stack in ("fr3", "franka_ros2", "fr3_pose"):
        return "fr3_link0"
    return "panda_link0"


def generate_launch_description():
    device     = LaunchConfiguration("device")
    width      = LaunchConfiguration("width")
    height     = LaunchConfiguration("height")
    framerate  = LaunchConfiguration("framerate")
    camera_url = LaunchConfiguration("camera_info")

    nodes = [
        DeclareLaunchArgument("device",      default_value="/dev/video0"),
        DeclareLaunchArgument("width",       default_value="1920"),
        DeclareLaunchArgument("height",      default_value="1080"),
        DeclareLaunchArgument("framerate",   default_value="15.0"),
        DeclareLaunchArgument("camera_info", default_value=f"file://{INTRINSICS}"),
    ]

    if os.path.exists(CALIB):
        args = load_camera_calibration(CALIB).static_transform_args(_base_frame(), CAMERA_FRAME)
        nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="camera_to_base_static_tf",
            arguments=["--x", args[0], "--y", args[1], "--z", args[2],
                       "--qx", args[3], "--qy", args[4], "--qz", args[5], "--qw", args[6],
                       "--frame-id", args[7], "--child-frame-id", args[8]]))
    else:
        print(f"[apriltag_webcam.launch] {CALIB} not found — skipping camera->base static "
              "transform; only camera->tag tf will be available until eye-to-hand "
              "calibration is run (scripts/calibrate_camera.py).")

    if not os.path.exists(INTRINSICS):
        print(f"[apriltag_webcam.launch] {INTRINSICS} not found — usb_cam will publish "
              "uncalibrated (zero) intrinsics until you run "
              "'ros2 run camera_calibration cameracalibrator' for this webcam.")

    nodes.extend([
        Node(package="usb_cam", executable="usb_cam_node_exe", name="camera", namespace="camera",
             parameters=[{
                 "video_device":    device,
                 "image_width":     width,
                 "image_height":    height,
                 "framerate":       framerate,
                 "pixel_format":    "mjpeg2rgb",
                 "camera_info_url": camera_url,
                 "frame_id":        CAMERA_FRAME,
                 # No SDK-managed TF; our eye-to-hand static tf covers base->camera.
                 "publish_camera_info_msg": True,
             }]),
        Node(package="apriltag_ros", executable="apriltag_node", name="apriltag",
             remappings=[("image_rect", CAMERA_IMAGE_TOPIC),
                         ("camera_info", CAMERA_INFO_TOPIC)],
             parameters=[APRILTAG_CFG]),
    ])

    return LaunchDescription(nodes)
