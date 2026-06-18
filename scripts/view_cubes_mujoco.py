#!/usr/bin/env python3
"""Live MuJoCo digital twin of the real Franka workspace.

Cubes are driven by AprilTag poses (tf2), the arm by /joint_states. The window is
READ-ONLY: it subscribes to /tf, /tf_static and /joint_states, publishes nothing to
control topics, and calls no state-mutating service -- so it is safe to run in
parallel with `make mile-real` / `make eval-real`. The loop only writes qpos and
calls mj_forward (kinematics, no physics), so nothing ever falls or is commanded.

Prereqs (real run): controller up (publishes /joint_states), `make apriltag-up`
(+ calibration for base->tag), host DISPLAY (`xhost +local:root`).

Usage (via `make view-twin`, or inside `make shell`):
    python3 scripts/view_cubes_mujoco.py
    python3 scripts/view_cubes_mujoco.py --joint-states-topic /panda/joint_states
    python3 scripts/view_cubes_mujoco.py --self-check    # no ROS/DISPLAY; scene+packing smoke
"""
import argparse
import time

import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE
from mile_franka.pose.base import Pose
from mile_franka.viz.mujoco_twin import (
    joint_writes, pose_to_freejoint_qpos, resolve_scene_path)

CUBES = (BOTTOM_CUBE, TOP_CUBE)


def _print_status(now, js_stamp, pose_src):
    parts = []
    parts.append("joints: " + ("n/a" if not js_stamp else f"{now - js_stamp:.1f}s"))
    for c in CUBES:
        seen = pose_src.last_seen(c)
        parts.append(f"{c}: " + ("never" if seen is None else f"{now - seen:.1f}s"))
    print("[twin] " + "  ".join(parts), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-states-topic", default="/joint_states",
                    help="JointState topic for the arm/gripper (default: /joint_states)")
    ap.add_argument("--rate", type=float, default=30.0, help="update rate Hz (default: 30)")
    ap.add_argument("--self-check", action="store_true",
                    help="load scene, place a cube at a known pose, verify, exit "
                         "(no ROS, no DISPLAY)")
    args = ap.parse_args()

    import mujoco

    cfg = StackTaskConfig()
    half_edge = cfg.cube_size / 2.0

    # ---- self-check: deterministic, no ROS/DISPLAY ----
    # Uses a minimal inline scene (cube free joints only) so this path runs without
    # ament/franka_description and with the mujoco 2.3.7 Python bindings that are
    # installed in the MILE conda env. The full scene (panda.xml + stacking_objects.xml)
    # is loaded only when launching the live viewer, where ROS is available anyway.
    if args.self_check:
        _SELF_CHECK_XML = """
<mujoco model="self-check">
  <worldbody>
    <body name="bottom_cube" pos="0.5 0.10 0.025">
      <geom name="bottom_cube" type="box" size="0.025 0.025 0.025"/>
      <joint name="bottom_cube_joint" type="free"/>
    </body>
    <body name="top_cube" pos="0.5 -0.10 0.025">
      <geom name="top_cube" type="box" size="0.025 0.025 0.025"/>
      <joint name="top_cube_joint" type="free"/>
    </body>
  </worldbody>
</mujoco>"""
        sc_model = mujoco.MjModel.from_xml_string(_SELF_CHECK_XML)
        sc_data = mujoco.MjData(sc_model)
        mujoco.mj_forward(sc_model, sc_data)

        # Verify pose packing: set bottom_cube to a known pose, read back xpos.
        jid = mujoco.mj_name2id(sc_model, mujoco.mjtObj.mjOBJ_JOINT, "bottom_cube_joint")
        a = int(sc_model.jnt_qposadr[jid])
        target = Pose(position=[0.5, 0.1, 0.3], orientation=[0.0, 0.0, 0.0, 1.0])
        sc_data.qpos[a:a + 7] = pose_to_freejoint_qpos(target)
        mujoco.mj_forward(sc_model, sc_data)
        bid = mujoco.mj_name2id(sc_model, mujoco.mjtObj.mjOBJ_BODY, BOTTOM_CUBE)
        np.testing.assert_allclose(sc_data.xpos[bid], [0.5, 0.1, 0.3], atol=1e-6)
        print("[twin] self-check OK (scene loads; base-frame pose == world pose)")
        return

    scene = resolve_scene_path()
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Free-joint qpos start address per cube.
    cube_qadr = {}
    for c in CUBES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{c}_joint")
        if jid < 0:
            raise RuntimeError(f"cube joint {c}_joint not found in {scene}")
        cube_qadr[c] = int(model.jnt_qposadr[jid])

    # ---- ROS ----
    import rclpy
    from sensor_msgs.msg import JointState
    from mile_franka.pose.apriltag import AprilTagPoseSource

    rclpy.init()
    node = rclpy.create_node("mile_franka_twin")

    js = {"names": [], "positions": [], "stamp": 0.0}

    def js_cb(msg):
        js["names"] = list(msg.name)
        js["positions"] = list(msg.position)
        js["stamp"] = time.time()

    node.create_subscription(JointState, args.joint_states_topic, js_cb, 10)
    pose_src = AprilTagPoseSource(half_edge=half_edge, node=node)

    import mujoco.viewer
    print(f"Scene        : {scene}")
    print(f"Joint topic  : {args.joint_states_topic}")
    print("Opening MuJoCo viewer (read-only twin). Close the window or Ctrl-C to stop.")
    viewer = mujoco.viewer.launch_passive(model, data)

    period = 1.0 / max(args.rate, 1.0)
    last_status = 0.0
    try:
        while rclpy.ok() and viewer.is_running():
            rclpy.spin_once(node, timeout_sec=period)

            for adr, val in joint_writes(model, js["names"], js["positions"]):
                data.qpos[adr] = val

            for c in CUBES:
                try:
                    pose = pose_src.get_pose(c)
                except RuntimeError:
                    continue  # never seen yet -> leave at rest
                a = cube_qadr[c]
                data.qpos[a:a + 7] = pose_to_freejoint_qpos(pose)

            mujoco.mj_forward(model, data)
            viewer.sync()

            now = time.time()
            if now - last_status >= 1.0:
                last_status = now
                _print_status(now, js["stamp"], pose_src)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
