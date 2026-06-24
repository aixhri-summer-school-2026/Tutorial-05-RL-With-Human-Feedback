# launch/apriltag_realsense.launch.py
"""Bring up the third-person RealSense, apriltag_ros, and (when calibrated) the camera->base
static transform.

Run on the same DDS graph as the controller so AprilTagPoseSource (tf2) can chain
base -> camera -> tag. Before calibration exists (config/camera_calib.yaml), the static
transform is skipped: realsense2_camera + apriltag_ros still publish camera -> tag tf, which
is enough to confirm detection. Once calibrate_camera.py has written camera_calib.yaml, the
static transform is added automatically and base -> tag resolves.

Args:
    serial_no: RealSense serial (default = the lab D415, 217222067236).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from mile_franka.pose.calibration import load_camera_calibration

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = (os.environ.get("MILE_CAMERA_CALIB")
         or os.path.join(REPO, "config", "camera_calib.yaml"))
APRILTAG_CFG = os.path.join(REPO, "config", "apriltag.yaml")
def _base_frame() -> str:
    real_stack = os.environ.get("MILE_REAL_STACK", "multipanda").lower()
    if real_stack in ("fr3", "franka_ros2", "fr3_pose"):
        return "fr3_link0"
    return "panda_link0"


CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_SERIAL = "217222067236"


def generate_launch_description():
    serial_no = LaunchConfiguration("serial_no")

    nodes = [
        DeclareLaunchArgument("serial_no", default_value=DEFAULT_SERIAL,
                              description="Intel RealSense serial number"),
    ]

    # Publish the camera→base static transform FIRST, before the camera and apriltag
    # nodes.  CycloneDDS+iceoryx shared memory can silently drop messages from publishers
    # that start after initial DDS discovery fills the shared-memory pool; placing the
    # one-shot static_transform_publisher ahead of the heavier nodes gives it the best
    # chance of securing a shared-memory slot and landing its /tf_static message.
    # AprilTagPoseSource also applies the calibration directly in Python (bypassing tf2),
    # so this transform is now a best-effort convenience for view-tags, view-twin, and
    # other tf2-only consumers.
    if os.path.exists(CALIB):
        args = load_camera_calibration(CALIB).static_transform_args(_base_frame(), CAMERA_FRAME)
        nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="camera_to_base_static_tf",
            arguments=["--x", args[0], "--y", args[1], "--z", args[2],
                       "--qx", args[3], "--qy", args[4], "--qz", args[5], "--qw", args[6],
                       "--frame-id", args[7], "--child-frame-id", args[8]]))
    else:
        print(f"[apriltag_realsense.launch] {CALIB} not found — skipping camera->base static "
              "transform; only camera->tag tf will be available until calibration is run.")

    nodes.extend([
        Node(package="realsense2_camera", executable="realsense2_camera_node", name="camera",
             # serial_no must be a STRING; an all-digit serial is otherwise parsed as an int
             # and realsense2_camera rejects it. Force the param type.
             # Color-only stream (depth + IR disabled): AprilTag detection needs only the color
             # image + camera_info. Run at 1080p — the D415 color sensor's max; on the USB 3.2
             # port (rs-enumerate-devices) there's no bandwidth constraint.
             # MEASURED 2026-06-18: the genuine AprilRobotics decoder in apriltag_ros needs the
             # tag at >=~120px to decode; the 36mm tags were ~75px at 720p (detected by OpenCV's
             # more-lenient aruco but NOT by apriltag_ros). 1080p puts them at ~106px — still
             # under the size threshold, so detection also depends on CONTRAST. Auto-exposure
             # meters the bright wood table and under-exposes the shadowed cubes (settled at
             # exposure~166, image mean~95 -> apriltag_ros decoded nothing). Forcing a fixed
             # higher exposure (mean ~150-195) made the genuine decoder detect BOTH tags at
             # decision_margin ~70. Exposure is in microseconds and is SCENE-DEPENDENT: retune
             # `rgb_camera.exposure` if the lighting changes (target image mean ~140-180; too
             # high washes the tag out). For robustness prefer larger (~45mm) tags + steady light.
             parameters=[{"serial_no": ParameterValue(serial_no, value_type=str),
                          "enable_color": True, "enable_depth": False,
                          "enable_infra1": False, "enable_infra2": False,
                          "rgb_camera.color_profile": "1920x1080x15",
                          "rgb_camera.enable_auto_exposure": False,
                          "rgb_camera.exposure": 1200,
                          "rgb_camera.gain": 80,
                          # Disable the camera's own TF tree (camera_link →
                          # camera_color_frame → camera_color_optical_frame).
                          # Our eye-to-hand static transform publishes
                          # fr3_link0 → camera_color_optical_frame directly,
                          # and tf2 rejects a frame with two parents, so the
                          # realsense TF would silently block the calibration.
                          "publish_tf": False}]),
        Node(package="apriltag_ros", executable="apriltag_node", name="apriltag",
             # realsense2_camera publishes on <camera_name>/color/... with camera_name
             # defaulting to the node name ("camera"), so the actual topics are
             # /camera/camera/color/image_raw and /camera/camera/color/camera_info.
             remappings=[("image_rect", "/camera/camera/color/image_raw"),
                         ("camera_info", "/camera/camera/color/camera_info")],
             parameters=[APRILTAG_CFG]),
    ])

    return LaunchDescription(nodes)
