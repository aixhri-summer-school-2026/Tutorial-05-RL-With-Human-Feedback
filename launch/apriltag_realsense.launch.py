# launch/apriltag_realsense.launch.py
"""Bring up the third-person RealSense, apriltag_ros, and the camera->base static transform.

Static transform args come from config/camera_calib.yaml (see CameraCalibration). Run on the
same DDS graph as the controller so AprilTagPoseSource (tf2) can chain base -> camera -> tag.
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node

from mile_franka.pose.calibration import load_camera_calibration

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(REPO, "config", "camera_calib.yaml")
APRILTAG_CFG = os.path.join(REPO, "config", "apriltag.yaml")
BASE_FRAME = "panda_link0"
CAMERA_FRAME = "camera_color_optical_frame"


def generate_launch_description():
    calib = load_camera_calibration(CALIB)
    static_tf_args = calib.static_transform_args(BASE_FRAME, CAMERA_FRAME)

    return LaunchDescription([
        Node(package="realsense2_camera", executable="realsense2_camera_node",
             name="camera", parameters=[{"enable_color": True, "enable_depth": False}]),
        Node(package="apriltag_ros", executable="apriltag_node", name="apriltag",
             remappings=[("image_rect", "/camera/color/image_raw"),
                         ("camera_info", "/camera/color/camera_info")],
             parameters=[APRILTAG_CFG]),
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="camera_to_base_static_tf",
             arguments=["--x", static_tf_args[0], "--y", static_tf_args[1],
                        "--z", static_tf_args[2], "--qx", static_tf_args[3],
                        "--qy", static_tf_args[4], "--qz", static_tf_args[5],
                        "--qw", static_tf_args[6], "--frame-id", static_tf_args[7],
                        "--child-frame-id", static_tf_args[8]]),
    ])
