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
CALIB = os.path.join(REPO, "config", "camera_calib.yaml")
APRILTAG_CFG = os.path.join(REPO, "config", "apriltag.yaml")
BASE_FRAME = "panda_link0"
CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_SERIAL = "217222067236"


def generate_launch_description():
    serial_no = LaunchConfiguration("serial_no")

    nodes = [
        DeclareLaunchArgument("serial_no", default_value=DEFAULT_SERIAL,
                              description="Intel RealSense serial number"),
        Node(package="realsense2_camera", executable="realsense2_camera_node", name="camera",
             # serial_no must be a STRING; an all-digit serial is otherwise parsed as an int
             # and realsense2_camera rejects it. Force the param type.
             # Color-only at a USB2-friendly profile: the camera enumerates on a USB 2.1 port,
             # so depth + both infrared streams starve the bus and frames time out. AprilTag
             # detection needs only the color image + camera_info.
             parameters=[{"serial_no": ParameterValue(serial_no, value_type=str),
                          "enable_color": True, "enable_depth": False,
                          "enable_infra1": False, "enable_infra2": False,
                          "rgb_camera.color_profile": "640x480x15"}]),
        Node(package="apriltag_ros", executable="apriltag_node", name="apriltag",
             # realsense2_camera publishes on <camera_name>/color/... with camera_name
             # defaulting to the node name ("camera"), so the actual topics are
             # /camera/camera/color/image_raw and /camera/camera/color/camera_info.
             remappings=[("image_rect", "/camera/camera/color/image_raw"),
                         ("camera_info", "/camera/camera/color/camera_info")],
             parameters=[APRILTAG_CFG]),
    ]

    # Only add the camera->base static transform once calibration exists; until then the
    # camera->tag tf alone confirms detection (see module docstring).
    if os.path.exists(CALIB):
        args = load_camera_calibration(CALIB).static_transform_args(BASE_FRAME, CAMERA_FRAME)
        nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="camera_to_base_static_tf",
            arguments=["--x", args[0], "--y", args[1], "--z", args[2],
                       "--qx", args[3], "--qy", args[4], "--qz", args[5], "--qw", args[6],
                       "--frame-id", args[7], "--child-frame-id", args[8]]))
    else:
        print(f"[apriltag_realsense.launch] {CALIB} not found — skipping camera->base static "
              "transform; only camera->tag tf will be available until calibration is run.")

    return LaunchDescription(nodes)
