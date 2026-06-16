from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.backend import RobotBackend

# CONFIRM@bringup: exact topic/action names exposed by the multipanda controller.
EQUILIBRIUM_TOPIC = "/cartesian_impedance/equilibrium_pose"
FRANKA_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
GRIPPER_ACTION = "/fr3_gripper/grasp"  # franka_msgs/action/Grasp
CONTROL_PERIOD_S = 0.1  # ~10 Hz control step (spec sec 5.2)


class MultipandaRosBackend(RobotBackend):
    """RobotBackend backed by a live multipanda_ros2 node (sim or hardware)."""

    def __init__(self, config: Optional[StackTaskConfig] = None,
                 base_frame: str = "fr3_link0"):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped

        self.config = config if config is not None else StackTaskConfig()
        self.base_frame = base_frame  # CONFIRM@bringup: robot base frame id

        if not rclpy.ok():
            rclpy.init()
        self._node: Node = rclpy.create_node("mile_franka_backend")
        self._pub = self._node.create_publisher(PoseStamped, EQUILIBRIUM_TOPIC, 10)

        self._ee_position = np.zeros(3, dtype=np.float32)
        self._gripper_width = float(self.config.gripper_open_width)
        self._lock = threading.Lock()
        self._subscribe_state()

        # Spin in a background thread so callbacks fire while the env runs.
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _subscribe_state(self) -> None:
        # CONFIRM@bringup: FrankaState message type + the field holding the EE pose.
        from franka_msgs.msg import FrankaRobotState

        def _on_state(msg) -> None:
            pose = msg.o_t_ee.pose  # CONFIRM@bringup: EE pose accessor
            with self._lock:
                self._ee_position = np.array(
                    [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)

        self._node.create_subscription(
            FrankaRobotState, FRANKA_STATE_TOPIC, _on_state, 10)

    def reset(self, np_random: np.random.Generator) -> None:
        # Real reset is human-gated (spec sec 5.2): place the cubes, then continue.
        input("Place the two cubes in the workspace and press Enter to start the episode...")

    def set_equilibrium_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        self._pub.publish(msg)
        time.sleep(CONTROL_PERIOD_S)  # wait one control period for the controller to track

    def set_gripper(self, command: float) -> None:
        # CONFIRM@bringup: gripper action interface (Grasp vs Move) + width/speed/force.
        from franka_msgs.action import Grasp
        from rclpy.action import ActionClient

        client = ActionClient(self._node, Grasp, GRIPPER_ACTION)
        if not client.wait_for_server(timeout_sec=2.0):
            return
        goal = Grasp.Goal()
        c = self.config
        goal.width = c.gripper_closed_width if command > 0 else c.gripper_open_width
        goal.speed = 0.1
        goal.force = 20.0
        client.send_goal_async(goal)  # fire-and-forget; controller closes/opens
        with self._lock:
            self._gripper_width = float(goal.width)

    def get_ee_position(self) -> np.ndarray:
        with self._lock:
            return self._ee_position.copy()

    def get_gripper_width(self) -> float:
        with self._lock:
            return float(self._gripper_width)

    def close(self) -> None:
        try:
            self._executor.shutdown()
            self._node.destroy_node()
        except Exception:
            pass
