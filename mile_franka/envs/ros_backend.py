"""RobotBackend backed by hucebot's multipanda_ros2 MuJoCo sim (or real hardware).

Verified against the running `hucebot:franka-humble` sim on 2026-06-16 (see the
franka-sim-verified-bringup note). Design notes:

- EE position is read from the controller's /cartesian_impedance/cartesian_pos_curr topic,
  which reports O_T_EE in the panda_link0 (base) frame -- the same frame the
  equilibrium_pose commands use. This is essential: get_body_state("panda_hand") returns
  the hand link which is ~0.103 m above the controller's EE frame, which would cause the
  scripted policy to command the arm underground before triggering GRASP.
- Cube poses (EE + cubes) are read from the mujoco_ros `get_body_state` SERVICE for cubes.
  The FrankaState broadcaster is not spawned by franka_sim_cartesian_impedance.launch.py.
  Cube poses are in the sim `world` frame, which coincides with `panda_link0` for the
  single arm modulo a 180-deg rotation about z (verified: link0 quat xyzw = [0,0,1,0]).
- The node is driven synchronously (call_async + spin_until_future_complete). We do NOT run
  a background spin thread, to avoid deadlocking service calls against a concurrent spinner.
- Gripper width is tracked from the last command (proxy); good enough for the obs and for
  the "released" success check. The Grasp goal is fired only when the open/closed state
  flips, and we wait only for goal acceptance (not completion) to keep step() ~10 Hz.

All ROS imports are lazy so the package imports with no ROS installed.
"""
from __future__ import annotations

import re
import time
import subprocess
from typing import Optional

import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.backend import RobotBackend

# Confirmed live names (hucebot:franka-humble, franka_sim_cartesian_impedance.launch.py).
EQUILIBRIUM_TOPIC = "/cartesian_impedance/equilibrium_pose"
EE_CURR_TOPIC = "/cartesian_impedance/cartesian_pos_curr"  # O_T_EE in panda_link0 frame
GET_BODY_STATE_SRV = "/get_body_state"
SET_BODY_STATE_SRV = "/set_body_state"
SET_PAUSE_SRV = "/set_pause"
GRASP_ACTION = "/panda_gripper_sim_node/grasp"  # franka_msgs/action/Grasp
EE_BODY = "panda_hand"  # used only if EE_CURR_TOPIC is unavailable
CONTROLLER_NAME = "custom_cartesian_impedance_controller"
MOVE_TO_START_CONTROLLER = "move_to_start_example_controller"
ENV_STEP_PERIOD_S = 0.1  # ~10 Hz Python env command/observation step
SIM_STACKING_GAINS = {
    "pos_stiff": 4000.0,
    "translational_clip": 0.5,
    "translational_Ki": 0.0,
    "rot_stiff": 800.0,
    "rotational_Ki": 0.0,
    "ns_stiff_q1_to_4": 0.0,
    "ns_stiff_q5_to_7": 0.0,
}


class MultipandaRosBackend(RobotBackend):
    """RobotBackend over a live multipanda_ros2 node (MuJoCo sim or hardware)."""

    def __init__(self, config: Optional[StackTaskConfig] = None,
                 base_frame: str = "panda_link0", ee_body: str = EE_BODY,
                 randomize_on_reset: bool = True,
                 controller_name: str = CONTROLLER_NAME,
                 move_to_start_controller: str = MOVE_TO_START_CONTROLLER,
                 activate_controller_on_reset: bool = True,
                 apply_sim_gains: bool = False,
                 reset_controller_target_on_reset: bool = False,
                 move_to_start_on_reset: bool = False,
                 env_step_period_s: float = ENV_STEP_PERIOD_S,
                 move_to_start_hold_s: float = 8.0,
                 home_steps: int = 80,
                 home_settle_s: float = 1.5):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from mujoco_ros_msgs.srv import GetBodyState, SetBodyState, SetPause
        from franka_msgs.action import Grasp
        from rclpy.action import ActionClient

        self.config = config if config is not None else StackTaskConfig()
        self.base_frame = base_frame
        self.ee_body = ee_body
        self.randomize_on_reset = randomize_on_reset
        self.controller_name = controller_name
        self.move_to_start_controller = move_to_start_controller
        self.activate_controller_on_reset = activate_controller_on_reset
        self.reset_controller_target_on_reset = reset_controller_target_on_reset
        self.move_to_start_on_reset = move_to_start_on_reset
        self.env_step_period_s = float(env_step_period_s)
        self.move_to_start_hold_s = float(move_to_start_hold_s)
        self.home_steps = int(home_steps)
        self.home_settle_s = float(home_settle_s)
        self._controller_activated_by_backend = False

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node = rclpy.create_node("mile_franka_backend")
        self._PoseStamped = PoseStamped
        self._SetBodyState = SetBodyState
        self._SetPause = SetPause
        self._Grasp = Grasp

        self._pub = self._node.create_publisher(PoseStamped, EQUILIBRIUM_TOPIC, 10)
        self._get_body = self._node.create_client(GetBodyState, GET_BODY_STATE_SRV)
        self._set_body = self._node.create_client(SetBodyState, SET_BODY_STATE_SRV)
        self._set_pause = self._node.create_client(SetPause, SET_PAUSE_SRV)
        self._grasp = ActionClient(self._node, Grasp, GRASP_ACTION)

        required_clients = [(self._get_body, GET_BODY_STATE_SRV),
                            (self._set_body, SET_BODY_STATE_SRV),
                            (self._set_pause, SET_PAUSE_SRV)]
        for client, name in required_clients:
            if not client.wait_for_service(timeout_sec=15.0):
                raise RuntimeError(f"multipanda service {name} not available")

        # Subscribe to the controller's EE pose (O_T_EE, already in panda_link0 frame).
        # Cached so get_ee_position() stays non-blocking.
        self._ee_curr_pose: Optional[np.ndarray] = None
        self._node.create_subscription(
            PoseStamped, EE_CURR_TOPIC,
            self._ee_curr_callback, 1)

        self._gripper_width = float(self.config.gripper_open_width)
        self._gripper_closed = False  # last commanded state
        # Always ensure the controllers we depend on are actually loaded before reset()
        # tries to activate them. On a slow cold boot the launch-time spawners can lose
        # the race against controller_manager and die (exit 1), leaving the controllers
        # never loaded -- self-heal that here rather than racing param-list.
        self._wait_for_controller_node()
        if apply_sim_gains:
            self._apply_controller_gains(SIM_STACKING_GAINS)

    # --- EE pose cache ---------------------------------------------------------
    def _ee_curr_callback(self, msg) -> None:
        """Cache the controller's current EE pose (panda_link0 frame, no flip needed)."""
        p = msg.pose.position
        self._ee_curr_pose = np.array([p.x, p.y, p.z], dtype=np.float32)

    def _spin_once_for_ee(self, timeout: float = 0.5) -> None:
        """Spin briefly to let the subscription callback fire and populate _ee_curr_pose."""
        deadline = time.time() + timeout
        while self._ee_curr_pose is None and time.time() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)

    # --- rclpy helpers ---------------------------------------------------------
    def _call(self, client, request, timeout: float = 5.0):
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout)
        if not future.done():
            raise RuntimeError("multipanda service call timed out")
        return future.result()

    def _run_ros2(self, args: list[str], *, timeout: float = 10.0,
                  check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["ros2", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"ros2 {' '.join(args)} failed: {detail}")
        return result

    def _list_controllers(self, timeout: float = 10.0):
        """Return {controller_name: state} from controller_manager, or None if unreachable.

        Queries controller_manager (`ros2 control list_controllers`) rather than the
        controller's own param node: a loaded controller and a never-loaded one are
        indistinguishable from `ros2 param list`, but controller_manager is the
        authoritative source and is up as soon as mujoco_server's ros2_control plugin is.
        """
        result = self._run_ros2(["control", "list_controllers"], timeout=timeout, check=False)
        if result.returncode != 0:
            return None
        # `ros2 control list_controllers` colorizes its output with ANSI escapes; strip them
        # so the controller name (parts[0]) and state (parts[-1]) parse cleanly.
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        states: dict[str, str] = {}
        for line in clean.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                states[parts[0]] = parts[-1].lower()
        return states

    def _load_controller(self, controller: str, state: str) -> None:
        """Load a controller into controller_manager at the given lifecycle state.

        The launch-time spawners can die on a slow cold boot before controller_manager is
        ready, leaving the controller never loaded. controller_manager already knows the
        controller's type from sim_stacking_controllers.yaml, so loading it here is
        idempotent recovery -- it self-heals a lost spawner race deterministically.
        """
        self._run_ros2(
            ["control", "load_controller", controller, "--set-state", state],
            timeout=20.0, check=False)

    def _wait_for_controller_node(self, timeout: float = 60.0) -> None:
        """Make the controllers we depend on deterministically ready.

        On a freshly-launched sim the controllers are spawned by the launch file, but on a
        slow software-GL cold boot those spawners can lose the race against
        controller_manager coming up and die (exit code 1), so the controllers are never
        loaded and the param node never appears ("Node not found"). Rather than racing, wait
        for controller_manager itself, then load any missing required controller via
        controller_manager (idempotent), and finally confirm the param interface is
        addressable so a subsequent `ros2 param set` of the gains will not race.
        """
        deadline = time.time() + timeout
        last = "controller_manager not reachable"
        # 1. Wait for controller_manager to answer, then ensure required controllers loaded.
        while time.time() < deadline:
            states = self._list_controllers()
            if states is not None:
                if states.get("joint_state_broadcaster") != "active":
                    self._load_controller("joint_state_broadcaster", "active")
                if self.controller_name not in states:
                    self._load_controller(self.controller_name, "inactive")
                # reset() activates move_to_start when move_to_start_on_reset; its spawner can
                # die in the same cold-boot race, so self-heal it too (idempotent, inactive).
                if (self.move_to_start_on_reset
                        and self.move_to_start_controller not in states):
                    self._load_controller(self.move_to_start_controller, "inactive")
                states = self._list_controllers() or {}
                ready = self.controller_name in states and (
                    not self.move_to_start_on_reset
                    or self.move_to_start_controller in states)
                if ready:
                    break
            last = "controller_manager up but %s not loaded" % self.controller_name
            time.sleep(1.0)
        else:
            raise RuntimeError(
                f"controller {self.controller_name} not loadable within {timeout:.0f}s: {last}")
        # 2. Confirm the controller's param interface is actually addressable.
        while time.time() < deadline:
            result = self._run_ros2(
                ["param", "list", f"/{self.controller_name}"], timeout=10.0, check=False)
            if result.returncode == 0:
                return
            last = (result.stderr or result.stdout).strip()
            time.sleep(1.0)
        raise RuntimeError(
            f"controller {self.controller_name} loaded but param node not addressable "
            f"within {timeout:.0f}s: {last}")

    def _apply_controller_gains(self, gains: dict[str, float]) -> None:
        """Apply the sim-validated stacking gains to the loaded controller."""
        for name, value in gains.items():
            self._run_ros2(
                ["param", "set", f"/{self.controller_name}", name, str(value)],
                timeout=5.0,
            )

    def _activate_controller(self) -> None:
        """Activate so the controller captures the current EE pose as its desired pose."""
        if not self.activate_controller_on_reset or self._controller_activated_by_backend:
            return
        result = self._run_ros2(
            ["control", "set_controller_state", self.controller_name, "active"],
            timeout=10.0,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode != 0 and "active" not in output:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"failed to activate {self.controller_name}: {detail}")
        self._controller_activated_by_backend = True

    def _set_controller_state(self, state: str, *, check: bool = False) -> None:
        self._set_named_controller_state(self.controller_name, state, check=check)

    def _set_named_controller_state(self, controller: str, state: str,
                                    *, check: bool = False) -> None:
        result = self._run_ros2(
            ["control", "set_controller_state", controller, state],
            timeout=10.0,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        if check and result.returncode != 0 and state not in output:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"failed to set {controller} {state}: {detail}")
        if controller == self.controller_name:
            self._controller_activated_by_backend = state == "active" and result.returncode == 0

    def _move_to_start(self) -> None:
        self._set_controller_state("inactive", check=False)
        self._set_named_controller_state(self.move_to_start_controller, "active", check=True)
        time.sleep(self.move_to_start_hold_s)
        self._set_named_controller_state(self.move_to_start_controller, "inactive", check=False)

    # sim reports poses in the `world` frame; the robot base (panda_link0) is at the world
    # origin but rotated 180 deg about z (verified: link0 quat xyzw = [0,0,1,0]). The whole
    # MILE env works in the base frame (matches the equilibrium-pose command frame and the
    # scripted policy's +x-forward convention), so transform every read into base and every
    # body-set back into world. The transform is its own inverse (180 about z).
    @staticmethod
    def _flip_xy_pos(p):
        return np.array([-p[0], -p[1], p[2]], dtype=np.float32)

    @staticmethod
    def _world_quat_to_base(q):  # xyzw world -> base, premultiply conj([0,0,1,0])
        x, y, z, w = q
        return np.array([y, -x, -w, z], dtype=np.float32)

    def get_body_pose(self, name: str) -> np.ndarray:
        """Return base-frame [x,y,z,qx,qy,qz,qw] for a sim body (via get_body_state)."""
        from mujoco_ros_msgs.srv import GetBodyState

        req = GetBodyState.Request()
        req.name = name
        res = self._call(self._get_body, req)
        if res is None or not res.success:
            raise RuntimeError(f"get_body_state failed for {name!r}")
        p = res.state.pose.pose.position
        o = res.state.pose.pose.orientation
        pos = self._flip_xy_pos([p.x, p.y, p.z])
        quat = self._world_quat_to_base([o.x, o.y, o.z, o.w])
        return np.concatenate([pos, quat]).astype(np.float32)

    def _unpause(self, attempts: int = 5) -> None:
        """Unpause the sim, verifying it took (clock advances) instead of fire-and-forget."""
        preq = self._SetPause.Request()
        preq.paused = False
        for _ in range(attempts):
            res = self._call(self._set_pause, preq)
            if res is not None and getattr(res, "success", True):
                return
            time.sleep(0.5)
        raise RuntimeError("failed to unpause sim via /set_pause (clock would stay frozen)")

    # --- RobotBackend ----------------------------------------------------------
    def reset(self, np_random: np.random.Generator) -> None:
        if self.reset_controller_target_on_reset:
            self._set_controller_state("inactive", check=False)

        # Unpause the (paused-on-boot) sim. The sim ignores SetPause until its ros2_control
        # plugin is fully up, so verify the response and retry rather than silently leaving
        # the clock frozen (a frozen clock means step() commands never take effect).
        self._unpause()

        self._gripper_closed = True  # force an open command even if our proxy is stale
        self._set_gripper(-1.0, wait_result=True, force=True)
        if self.move_to_start_on_reset:
            self._move_to_start()

        if self.randomize_on_reset:
            c = self.config
            lo = c.workspace_low[:2] + c.reset_margin
            hi = c.workspace_high[:2] - c.reset_margin
            while True:
                bottom_xy = np_random.uniform(lo, hi)
                top_xy = np_random.uniform(lo, hi)
                if np.linalg.norm(bottom_xy - top_xy) >= c.reset_min_separation:
                    break
            rest_z = c.table_z + c.cube_size / 2.0
            self._set_body_pose_verified("bottom_cube", [*bottom_xy, rest_z])
            self._set_body_pose_verified("top_cube", [*top_xy, rest_z])
        self._home(activate_controller=True)

    def _home(self, activate_controller: bool = False) -> None:
        """Drive the arm to a consistent reachable start pose and let it settle.

        The multipanda custom Cartesian controller sets its desired pose to the current EE
        pose in `on_activate()`. Cycling inactive->active is therefore the stack-native way
        to discard a stale equilibrium target. After activation, command home through the
        controller; do not use MuJoCo `/reset` for the arm in ROS 2, because the generic
        initial-joint loader is NYI and the ros2_control plugin reset is a no-op.
        """
        from mile_franka.config import DOWN_QUAT

        c = self.config
        home = np.array([0.45, 0.0, c.table_z + 0.33], dtype=np.float32)
        if activate_controller:
            self._activate_controller()

        start = self.get_ee_position()
        for alpha in np.linspace(0.0, 1.0, self.home_steps):
            target = (1.0 - alpha) * start + alpha * home
            self._publish_pose(target, DOWN_QUAT)
            self._rclpy.spin_once(self._node, timeout_sec=0.02)
            time.sleep(0.03)
        t = time.time()
        while time.time() - t < self.home_settle_s:
            self._publish_pose(home, DOWN_QUAT)
            self._rclpy.spin_once(self._node, timeout_sec=0.02)
            time.sleep(0.03)
        self._ee_curr_pose = None
        self._spin_once_for_ee()

    def _set_body_pose(self, name: str, position) -> None:
        """position is base-frame; the service sets in world, so flip x,y back to world."""
        from mujoco_ros_msgs.msg import BodyState

        world = self._flip_xy_pos(position)
        req = self._SetBodyState.Request()
        bs = BodyState()
        bs.name = name
        bs.pose.header.frame_id = "world"
        bs.pose.pose.position.x = float(world[0])
        bs.pose.pose.position.y = float(world[1])
        bs.pose.pose.position.z = float(world[2])
        bs.pose.pose.orientation.w = 1.0
        req.state = bs
        req.set_pose = True
        req.set_twist = False
        req.set_mass = False
        req.reset_qpos = False
        self._call(self._set_body, req)

    def _set_body_pose_verified(self, name: str, position,
                                tol: float = 0.015,
                                attempts: int = 5) -> None:
        """Set a body pose and verify get_body_state has converged before reset returns."""
        target = np.asarray(position, dtype=np.float32)
        last = None
        for _ in range(attempts):
            self._set_body_pose(name, target)
            time.sleep(0.1)
            last = self.get_body_pose(name)[:3]
            if float(np.linalg.norm(last - target)) < tol:
                return
        raise RuntimeError(
            f"failed to reset {name} near {target.tolist()}; last pose was "
            f"{None if last is None else last.tolist()}"
        )

    def _publish_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        msg = self._PoseStamped()
        msg.header.frame_id = self.base_frame  # = panda_link0
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        self._pub.publish(msg)

    def set_equilibrium_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        target = np.asarray(position, dtype=np.float32)

        # Keep the policy/env contract at 10 Hz: one policy action maps to one Cartesian
        # target. The Cartesian controller tracks and holds that setpoint internally at its
        # own control frequency until the next policy tick.
        self._ee_curr_pose = None  # invalidate so next get_ee_position() waits for fresh data
        deadline = time.time() + self.env_step_period_s
        self._publish_pose(target, orientation)
        while time.time() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=min(0.01, deadline - time.time()))

    def _set_gripper(self, command: float, wait_result: bool = False,
                     force: bool = False) -> None:
        want_closed = command > 0
        c = self.config
        if want_closed == self._gripper_closed and not force:
            return  # no state change; don't spam the action server
        self._gripper_closed = want_closed
        grasp_width = min(c.gripper_open_width, max(c.gripper_closed_width, 0.95 * c.cube_size))
        self._gripper_width = grasp_width if want_closed else c.gripper_open_width
        if not self._grasp.wait_for_server(timeout_sec=2.0):
            return
        goal = self._Grasp.Goal()
        goal.width = float(self._gripper_width)
        goal.speed = 0.1
        goal.force = 80.0   # increased from 40 N — cube was slipping under light grip
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.005
        future = self._grasp.send_goal_async(goal)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=2.0)
        if not wait_result or not future.done():
            return
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            return
        result_future = goal_handle.get_result_async()
        self._rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=4.0)

    def set_gripper(self, command: float) -> None:
        # Wait for both close and open to complete — without waiting on close the EE
        # moves before the gripper has physically shut, causing the cube to slip.
        self._set_gripper(command, wait_result=True)

    def get_ee_position(self) -> np.ndarray:
        """Return the controller's O_T_EE position (panda_link0 frame).

        Reads from /cartesian_impedance/cartesian_pos_curr, which is already in the same
        frame as the equilibrium_pose commands. Falls back to get_body_state("panda_hand")
        minus the hand-to-flange offset if the topic has not published yet.
        """
        if self._ee_curr_pose is None:
            self._spin_once_for_ee()
        if self._ee_curr_pose is not None:
            return self._ee_curr_pose.copy()
        # Fallback: panda_hand is ~0.103 m above the controller EE frame.
        return self.get_body_pose(self.ee_body)[:3] - np.array([0.0, 0.0, 0.103],
                                                                dtype=np.float32)

    def get_gripper_width(self) -> float:
        return float(self._gripper_width)

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        # Shut down the rclpy context so the next env construction can reinitialise cleanly.
        # The init guard (rclpy.ok() check in __init__) handles the restart side.
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
