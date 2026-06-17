"""Waypoint grasp/stack demo for visual review (runs in the multipanda container).

Assumes the sim was launched with mile_franka/launch/franka_sim_stacking.launch.py, which
loads the cartesian impedance controller **inactive**. This demo:
  1. sets firm controller gains,
  2. reads the home EE pose (TF) while the controller is inactive (arm at hardware home),
  3. publishes the home pose as the equilibrium target, then ACTIVATES the controller
     (so it never dives to the origin -> arm stays at the clean home config),
  4. runs absolute-waypoint grasp+stack with the gripper pointing straight down,
  5. records the headless viewer to MP4.

For normal MILE collection/inference, `MultipandaRosBackend.reset()` owns the same
home-prime -> activate sequence. This script keeps the sequence inline because it is a
standalone visual sanity check, not a `FrankaEnv` rollout.
"""
import subprocess, time
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from mujoco_ros_msgs.srv import GetBodyState, SetPause, SetBodyState
from mujoco_ros_msgs.msg import BodyState
from franka_msgs.action import Grasp
from rclpy.action import ActionClient
import tf2_ros
from mile_franka.envs.ros_backend import CONTROLLER_NAME, SIM_STACKING_GAINS

DOWN = [1.0, 0.0, 0.0, 0.0]              # straight-down gripper (xyzw), base frame
def w2b(p): return [-p[0], -p[1], p[2]]  # base<->world (180 about z)
CTRL = CONTROLLER_NAME
GAINS = SIM_STACKING_GAINS
PUB_DT = 0.04


def min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def main():
    for k, v in GAINS.items():
        subprocess.run(["ros2", "param", "set", "/" + CTRL, k, str(v)], stdout=subprocess.DEVNULL)
    rclpy.init(); n = rclpy.create_node("grasp_demo")
    sp = n.create_client(SetPause, "/set_pause"); sp.wait_for_service()
    sb = n.create_client(SetBodyState, "/set_body_state"); sb.wait_for_service()
    gb = n.create_client(GetBodyState, "/get_body_state"); gb.wait_for_service()
    pub = n.create_publisher(PoseStamped, "/cartesian_impedance/equilibrium_pose", 10)
    grasp = ActionClient(n, Grasp, "/panda_gripper_sim_node/grasp")
    buf = tf2_ros.Buffer(); tf2_ros.TransformListener(buf, n)
    def call(c, r): f = c.call_async(r); rclpy.spin_until_future_complete(n, f); return f.result()
    call(sp, SetPause.Request(paused=False))

    t = time.time()
    while time.time() - t < 4: rclpy.spin_once(n, timeout_sec=0.05)   # warm up TF
    try:
        home = buf.lookup_transform(
            "panda_link0", "panda_hand_tcp", rclpy.time.Time()).transform.translation
        HOME = [home.x, home.y, home.z]
    except Exception:
        req = GetBodyState.Request(); req.name = "panda_hand"
        res = call(gb, req)
        p = res.state.pose.pose.position
        HOME = w2b([p.x, p.y, p.z])
    print("home EE:", np.round(HOME, 3))

    def tcp():
        tr = buf.lookup_transform(
            "panda_link0", "panda_hand_tcp", rclpy.time.Time()).transform.translation
        return np.array([tr.x, tr.y, tr.z], dtype=float)
    def cube(name):
        req = GetBodyState.Request(); req.name = name
        res = call(gb, req)
        p = res.state.pose.pose.position
        return np.array(w2b([p.x, p.y, p.z]), dtype=float)
    def trace(label):
        try:
            print(label, "tcp", np.round(tcp(), 3),
                  "top", np.round(cube("top_cube"), 3),
                  "bottom", np.round(cube("bottom_cube"), 3))
        except Exception as exc:
            print(label, "trace_failed", exc)
    def publish_pose(pos):
        m = PoseStamped(); m.header.frame_id = "panda_link0"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = DOWN
        pub.publish(m)
    def settle(pos, hold=0.8):
        t = time.time()
        while time.time() - t < hold:
            publish_pose(pos); rclpy.spin_once(n, timeout_sec=0.01); time.sleep(PUB_DT)
    def settle_until(pos, tol=0.018, min_s=0.8, max_s=4.0):
        target = np.array(pos, dtype=float)
        t0 = time.time()
        while time.time() - t0 < max_s:
            publish_pose(target)
            rclpy.spin_once(n, timeout_sec=0.01)
            err = float(np.linalg.norm(tcp() - target))
            if time.time() - t0 >= min_s and err < tol:
                break
            time.sleep(PUB_DT)
    def move(pos, duration=5.0, settle_s=1.5, tol=0.018):
        start = tcp()
        target = np.array(pos, dtype=float)
        steps = max(2, int(duration / PUB_DT))
        for i in range(steps + 1):
            s = min_jerk(i / steps)
            publish_pose((1.0 - s) * start + s * target)
            rclpy.spin_once(n, timeout_sec=0.01)
            time.sleep(PUB_DT)
        settle_until(target, tol=tol, min_s=settle_s, max_s=max(3.0, settle_s + 3.0))
    def gripper(width):
        if not grasp.wait_for_server(timeout_sec=2.0): return
        g = Grasp.Goal(); g.width = float(width); g.speed = 0.1; g.force = 40.0
        g.epsilon.inner = 0.08; g.epsilon.outer = 0.08
        f = grasp.send_goal_async(g); rclpy.spin_until_future_complete(n, f); time.sleep(1.5)
    def place_cube(name, base_xyz):
        bs = BodyState(); bs.name = name; w = w2b(base_xyz)
        bs.pose.pose.position.x, bs.pose.pose.position.y, bs.pose.pose.position.z = map(float, w)
        bs.pose.pose.orientation.w = 1.0
        req = SetBodyState.Request(); req.state = bs; req.set_pose = True
        call(sb, req)

    # publish home target, then ACTIVATE the controller (no dive)
    subprocess.run(["ros2", "control", "set_controller_state", CTRL, "inactive"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        publish_pose(HOME); time.sleep(PUB_DT)
    subprocess.run(["ros2", "control", "set_controller_state", CTRL, "active"], stdout=subprocess.PIPE)
    settle(HOME, 1.5)

    bottom = [0.45, 0.12, 0.031]; top = [0.45, -0.12, 0.031]
    place_cube("bottom_cube", bottom); place_cube("top_cube", top)
    time.sleep(1.0)
    trace("placed")

    rec = subprocess.Popen(["ffmpeg","-y","-f","x11grab","-video_size","1280x720","-framerate","15",
                            "-i",":99","-pix_fmt","yuv420p","/tmp/grasp_demo.mp4"],
                           stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    try:
        gripper(0.08)                                   # open
        move([top[0], top[1], 0.30], duration=5.0)      # hover over top cube
        trace("hover_pick")
        move([top[0], top[1], 0.035], duration=6.0, settle_s=2.0)  # descend to cube center
        trace("descend_pick")
        gripper(0.0)                                     # grasp
        trace("after_grasp")
        move([top[0], top[1], 0.30], duration=5.0)      # lift
        trace("lift")
        move([bottom[0], bottom[1], 0.30], duration=5.0)  # transit over bottom cube
        trace("hover_place")
        move([bottom[0], bottom[1], 0.095], duration=7.0, settle_s=3.0, tol=0.02)  # lower to one cube height
        trace("descend_place")
        gripper(0.08)                                    # release
        trace("after_release")
        move([bottom[0], bottom[1], 0.30], duration=4.0)  # retreat
        trace("retreat")
    finally:
        try: rec.communicate(input=b"q", timeout=10)
        except Exception: rec.terminate()
    n.destroy_node(); rclpy.shutdown()
    print("done -> /tmp/grasp_demo.mp4")


if __name__ == "__main__":
    main()
