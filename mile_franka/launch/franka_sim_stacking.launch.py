"""Launch the multipanda MuJoCo sim with the MILE stacking scene + cartesian impedance.

Based on franka_bringup/launch/sim/franka_sim_cartesian_impedance.launch.py, with three
changes for the MILE task:
  - modelfile points at our stacking scene (cubes), passed via `scene_path`;
  - no RViz / interactive marker;
  - the move-to-start and Cartesian-impedance controllers are spawned **inactive**, so
    reset can explicitly choose joint-space recovery before Cartesian control.

This launch is not the long-term "move home" abstraction. Homing belongs in the robot
backend reset path so demo collection and learned-policy inference use the same safe
startup sequence. The inactive spawn is only the lifecycle precondition that lets the
backend run joint-space recovery when needed, activate the Cartesian controller from the
current EE pose, discard stale targets, and then command a bounded Cartesian move to home.

The scene file must sit beside franka_description's panda.xml at runtime (relative includes);
the run script copies it into that share dir and passes its path as `scene_path`.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    fd = get_package_share_directory("franka_description")
    default_scene = os.path.join(fd, "mujoco", "franka", "scene.xml")
    xacro_file = os.path.join(fd, "robots", "sim", "panda_arm_sim.urdf.xacro")
    mjros_cfg = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "config", "sim_stacking_controllers.yaml"))
    bringup = get_package_share_directory("franka_bringup")

    arm_id = LaunchConfiguration("arm_id")
    initial_positions = LaunchConfiguration("initial_positions")
    scene_path = LaunchConfiguration("scene_path")

    robot_description = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " arm_id:=", arm_id, " hand:=true", " initial_positions:=", initial_positions])

    return LaunchDescription([
        DeclareLaunchArgument("arm_id", default_value="panda"),
        # Franka "ready"/home config (move_to_start q_goal in franka_example_controllers).
        DeclareLaunchArgument(
            "initial_positions",
            default_value='"-0.008 -0.005 0.011 -1.563 0.005 1.603 0.850"'),
        DeclareLaunchArgument("scene_path", default_value=default_scene),
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(bringup + "/launch/sim/launch_mujoco_ros_server.launch"),
            launch_arguments={"use_sim_time": "true", "modelfile": scene_path,
                              "verbose": "true", "ns": "",
                              "mujoco_plugin_config": mjros_cfg}.items()),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[{"robot_description": robot_description}]),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             name="joint_state_publisher",
             parameters=[{"source_list": ["/joint_state_broadcaster/joint_states",
                                          "/panda_gripper_sim_node/joint_states"],
                          "rate": 30}]),
        # Controller loading is handled by MultipandaRosBackend._wait_for_controller_node()
        # (mile_franka/envs/ros_backend.py). The launch-time spawners reliably lose the race
        # against controller_manager warm-up (~20 s on cold boot before service callbacks are
        # processed) and die with exit code 1, taking the whole launch down.
        # _wait_for_controller_node polls controller_manager with up to a 60 s deadline,
        # then self-heals any missing controller (joint_state_broadcaster → active,
        # custom_cartesian_impedance_controller → inactive, move_to_start_example_controller
        # → inactive), so the sim launches fast and the Python client handles the warm-up.
    ])
