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
import os
import time

import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE
from mile_franka.pose.base import Pose
from mile_franka.viz.mujoco_twin import (
    base_pose_to_world_qpos, joint_writes, pose_to_freejoint_qpos,
    resolve_scene_path)

CUBES = (BOTTOM_CUBE, TOP_CUBE)


def _default_base_frame() -> str:
    real_stack = os.environ.get("MILE_REAL_STACK", "multipanda").lower()
    if real_stack in ("fr3", "franka_ros2", "fr3_pose"):
        return "fr3_link0"
    return "panda_link0"


def _print_status(now, js, pose_src, poses, label=""):
    """Print one-line status: joint ages, cube ages+positions, and arm joint angles (deg)."""
    parts = [f"[{label}]"] if label else ["[twin]"]
    js_stamp = js.get("stamp", 0.0)
    parts.append("js:" + ("n/a" if not js_stamp else f"{now - js_stamp:.1f}s"))
    for c in CUBES:
        seen = pose_src.last_seen(c)
        age_str = "never" if seen is None else f"{now - seen:.1f}s"
        p = poses.get(c)
        if p is not None:
            xyz = f"({p.position[0]:+.3f} {p.position[1]:+.3f} {p.position[2]:+.3f})"
            parts.append(f"{c}:{xyz} {age_str}")
        else:
            parts.append(f"{c}:{age_str}")

    # Compact arm-joint row: j1–j7 in degrees, plus grip width (mm).
    if js.get("positions") and js.get("names"):
        # Match the seven arm joints in order (panda_joint1..7 or franka_joint1..7).
        arm = []
        grip = None
        for name, pos in zip(js["names"], js["positions"]):
            if "finger" in name:
                grip = pos if grip is None else max(grip, pos)
            elif "joint" in name and len(arm) < 7:
                arm.append((name, pos))
        arm.sort(key=lambda kv: kv[0])  # sort by name so j1..j7 come out in order
        if len(arm) == 7:
            degs = " ".join(f"{np.degrees(v):+6.1f}°" for _, v in arm)
            parts.append(f"arm:[{degs} ]")
        if grip is not None:
            parts.append(f"grip:{grip*1000:.0f}mm")
    print("  ".join(parts), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-states-topic", default="/joint_states",
                    help="JointState topic for the arm/gripper (default: /joint_states)")
    ap.add_argument("--rate", type=float, default=30.0, help="update rate Hz (default: 30)")
    ap.add_argument("--base-frame", default=None,
                    help="robot base frame for AprilTag tf lookups "
                         "(default: from MILE_REAL_STACK env — fr3_link0 / panda_link0)")
    ap.add_argument("--self-check", action="store_true",
                    help="load scene, place a cube at a known pose, verify, exit "
                         "(no ROS, no DISPLAY)")
    args = ap.parse_args()

    import mujoco  # twin viewer runs on mujoco 2.3.7 in the tutorial image (see Dockerfile)

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

    # Cube poses arrive in the ROBOT BASE frame (panda_link0). The franka MJCF
    # places panda_link0 with a non-identity world transform (quat (wxyz)
    # [0,0,0,1] = 180 deg about Z), so base coords must be composed with the base
    # body's world pose before they go into world free-joint qpos -- otherwise the
    # cubes render rotated about the base (behind the arm, mirrored). The base
    # body is static (root, no free joint), so we read its world transform once.
    _base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_link0")
    if _base_bid < 0:
        raise RuntimeError(f"panda_link0 body not found in {scene}")
    base_xpos = data.xpos[_base_bid].copy()
    base_xquat = data.xquat[_base_bid].copy()  # MuJoCo wxyz

    # ---- ROS ----
    import rclpy
    from sensor_msgs.msg import JointState
    from mile_franka.pose.apriltag import AprilTagPoseSource
    from mile_franka.pose.calibration import load_camera_calibration

    rclpy.init()
    node = rclpy.create_node("mile_franka_twin")

    js = {"names": [], "positions": [], "stamp": 0.0}
    robot_prefix = "panda"  # default; auto-detected from first joint_state

    def js_cb(msg):
        nonlocal robot_prefix
        js["names"] = list(msg.name)
        js["positions"] = list(msg.position)
        js["stamp"] = time.time()
        # Auto-detect robot type from first joint name. Both the multipanda sim
        # (franka_*) and the real franka_ros2 FR3 stack (fr3_*) use fr3_link0 as
        # their base frame; only the bare multipanda demo uses panda_*.
        if js["names"] and robot_prefix == "panda":
            for n in js["names"]:
                if n.startswith("franka_") or n.startswith("fr3_"):
                    robot_prefix = "franka"
                    break

    node.create_subscription(JointState, args.joint_states_topic, js_cb, 10)

    # Spin briefly to receive first joint_states for auto-detection.
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.02)
        if js["names"]:
            break

    # Determine base frame from robot type (scene always uses panda_* names).
    if args.base_frame:
        base_frame = args.base_frame
    elif robot_prefix == "franka":
        base_frame = "fr3_link0"
    else:
        base_frame = "panda_link0"

    # Joint name map: /joint_states names -> MuJoCo scene joint names.
    # The scene always uses panda_joint*; the multipanda sim publishes franka_joint*
    # and the real franka_ros2 FR3 stack publishes fr3_joint*.
    _joint_name_map = {}
    for i in range(1, 8):
        _joint_name_map[f"franka_joint{i}"] = f"panda_joint{i}"
        _joint_name_map[f"fr3_joint{i}"] = f"panda_joint{i}"
        _joint_name_map[f"panda_joint{i}"] = f"panda_joint{i}"
    # Finger joints (panda / franka / fr3 prefixes all map to the scene's panda_*).
    for prefix in ("panda", "franka", "fr3"):
        _joint_name_map[f"{prefix}_finger_joint1"] = "panda_finger_joint1"
        _joint_name_map[f"{prefix}_finger_joint2"] = "panda_finger_joint2"

    # ---- Load camera→base extrinsics (direct composition, no /tf_static) ----
    # The AprilTagPoseSource now accepts a camera_to_base 4×4 and composes
    # base_T_tag = camera_to_base @ camera_T_tag in Python, bypassing the
    # /tf_static chain entirely (fragile with CycloneDDS shared-memory
    # discovery).  We still publish the static transform for other consumers
    # (view-tags before the direct-composition fix, rviz, etc.).
    calib_path = os.environ.get("MILE_CAMERA_CALIB",
                                 os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              "config", "camera_calib.yaml"))
    _camera_to_base = None  # 4×4 or None
    if os.path.isfile(calib_path):
        try:
            calib = load_camera_calibration(calib_path)
            _camera_to_base = calib.camera_to_base
            from geometry_msgs.msg import TransformStamped
            import tf2_ros
            calib_tf = TransformStamped()
            calib_tf.header.stamp = node.get_clock().now().to_msg()
            calib_tf.header.frame_id = base_frame
            calib_tf.child_frame_id = "camera_color_optical_frame"
            t = calib.camera_to_base[:3, 3]
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(calib.camera_to_base[:3, :3]).as_quat()  # xyzw
            calib_tf.transform.translation.x = float(t[0])
            calib_tf.transform.translation.y = float(t[1])
            calib_tf.transform.translation.z = float(t[2])
            calib_tf.transform.rotation.x = float(quat[0])
            calib_tf.transform.rotation.y = float(quat[1])
            calib_tf.transform.rotation.z = float(quat[2])
            calib_tf.transform.rotation.w = float(quat[3])
            calib_broadcaster = tf2_ros.StaticTransformBroadcaster(node)
            calib_broadcaster.sendTransform(calib_tf)
            print(f"[twin] loaded camera→base from {calib_path}"
                  f"  (static tf: {base_frame} → camera_color_optical_frame)")
        except Exception as e:
            print(f"[twin] calibration load/publish failed: {e} (path={calib_path})")
    else:
        print(f"[twin] no calibration at {calib_path} — cube poses will need GT source")

    # ---- Cube pose source: try GT (sim) first, fall back to AprilTag (real) ----
    pose_src = None
    pose_src_label = ""

    # Attempt GT poses via mujoco_ros /get_body_state service (sim only).
    try:
        from mujoco_ros_msgs.srv import GetBodyState
        gt_client = node.create_client(GetBodyState, "/get_body_state")
        if gt_client.wait_for_service(timeout_sec=0.3):
            # Test with one known body to confirm the service works.
            for test_name in ("bottom_cube", "top_cube"):
                req = GetBodyState.Request()
                req.name = test_name
                future = gt_client.call_async(req)
                rclpy.spin_until_future_complete(node, future, timeout_sec=0.5)
                if future.result() is not None and future.result().success:
                    pose_src_label = "mujoco_gt"
                    break
            if pose_src_label:
                print(f"[twin] using GT cube poses from /get_body_state (sim)")
            else:
                node.destroy_client(gt_client)
        else:
            node.destroy_client(gt_client)
    except ImportError:
        pass

    if pose_src_label == "mujoco_gt":
        # Inline GT pose source wrapping the service client.
        class _GtSource:
            def __init__(self, node, client):
                self._node = node
                self._client = client
                self._last = {}
            def get_pose(self, name):
                from mile_franka.pose.base import Pose
                req = GetBodyState.Request()
                req.name = name
                future = self._client.call_async(req)
                rclpy.spin_until_future_complete(self._node, future, timeout_sec=0.5)
                result = future.result()
                if result is None or not result.success:
                    raise RuntimeError(f"get_body_state failed for {name!r}")
                p = result.state.pose
                self._last[name] = time.time()
                return Pose((p.position.x, p.position.y, p.position.z),
                            (p.orientation.x, p.orientation.y,
                             p.orientation.z, p.orientation.w))
            def last_seen(self, name):
                return self._last.get(name)
        pose_src = _GtSource(node, gt_client)
    else:
        # AprilTag: auto-detect which base frame actually has a calibration path
        # to the tags. The calibration static transform is published with a base
        # frame that depends on MILE_REAL_STACK (fr3_link0 vs panda_link0), and
        # if it doesn't match the robot's tf tree the chain is disconnected.
        # Try each candidate; use the first one that resolves a tag.
        candidates = [base_frame]
        if base_frame == "panda_link0":
            candidates.append("fr3_link0")
        else:
            candidates.append("panda_link0")

        chosen_base = base_frame
        for bf in candidates:
            try:
                src = AprilTagPoseSource(half_edge=half_edge, node=node,
                                          base_frame=bf,
                                          camera_to_base=_camera_to_base)
                # Spin a bit to let the tf listener prime
                for _ in range(30):
                    rclpy.spin_once(node, timeout_sec=0.05)
                src.get_pose("bottom_cube")
                chosen_base = bf
                pose_src = src
                break
            except RuntimeError:
                continue
        else:
            # None worked; use the original base_frame so the user sees the error.
            pose_src = AprilTagPoseSource(half_edge=half_edge, node=node,
                                           base_frame=base_frame,
                                           camera_to_base=_camera_to_base)
        base_frame = chosen_base
        pose_src_label = "apriltag"

    print(f"Base frame   : {base_frame}")
    print(f"Robot prefix : {robot_prefix}")
    print(f"Pose source  : {pose_src_label}")

    import mujoco.viewer  # twin viewer runs on mujoco 2.3.7 in the tutorial image (see Dockerfile)
    print(f"Scene        : {scene}")
    print(f"Joint topic  : {args.joint_states_topic}")

    # NOTE: do NOT enable mjVIS_JOINT here. It renders MuJoCo's built-in joint
    # markers, whose default colour is cyan (vis.rgba.joint = 0.2,0.6,0.8) and
    # whose size scales with the model extent. The two cubes have *free* joints,
    # so the flag draws a large cyan marker centred on each cube -- hiding the
    # real 5 cm blue/red cube geoms and making them look big and colourless.
    # Arm joint configuration is already shown by the per-link coloured spheres
    # drawn in the loop below, so the built-in markers are redundant anyway.
    # launch_passive's `show_left_ui` kwarg needs mujoco>=3.0; the twin is pinned
    # to 2.3.7 (see docker/requirements-mile.txt), whose Handle has no programmatic
    # panel toggle. Left tabbed settings panel starts visible -- hide it in-viewer
    # with Tab if it's in the way.
    viewer = mujoco.viewer.launch_passive(model, data)

    # Collect link body ids and joint qpos addresses for in-viewer joint markers.
    LINK_BODIES = [f"panda_link{i}" for i in range(1, 9)]  # link1..link8
    _link_body_id = {}
    _link_qadr = {}
    for i, body_name in enumerate(LINK_BODIES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            _link_body_id[body_name] = bid
        # Corresponding joint: panda_joint1..panda_joint7
        if i < 7:
            jname = f"panda_joint{i+1}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                _link_qadr[jname] = int(model.jnt_qposadr[jid])

    _last_pose = {}

    period = 1.0 / max(args.rate, 1.0)
    last_status = 0.0
    try:
        while rclpy.ok() and viewer.is_running():
            # Drain all pending messages before this frame (same pattern as
            # view_camera_tags.py).  spin_once processes only one message;
            # /joint_states and /tf messages from apriltag_ros compete for
            # slots — draining guarantees both are consumed.
            drain_deadline = time.time() + max(period, 0.05)
            while time.time() < drain_deadline:
                rclpy.spin_once(node, timeout_sec=0.005)

            # Map /joint_states names to scene joint names (e.g. franka_joint1 -> panda_joint1).
            scene_names = [_joint_name_map.get(n, n) for n in js["names"]]
            for adr, val in joint_writes(model, scene_names, js["positions"]):
                data.qpos[adr] = val

            for c in CUBES:
                try:
                    pose = pose_src.get_pose(c)
                except RuntimeError:
                    continue  # never seen yet -> leave at rest
                a = cube_qadr[c]
                # pose is in the robot base frame; compose with the base body's
                # world transform before writing world free-joint qpos.
                data.qpos[a:a + 7] = base_pose_to_world_qpos(
                    pose, base_xpos, base_xquat)
                # Store resolved pose for status display (base-frame coords).
                _last_pose[c] = pose

            mujoco.mj_forward(model, data)

            # ---- in-viewer joint-config markers ----
            # `user_scn` (a scene layer mj_forward/mjv_updateScene never touches,
            # so markers persist across frames) needs mujoco>=3.0; the twin is
            # pinned to 2.3.7 (see docker/requirements-mile.txt), whose Handle
            # has no such layer. Skip the overlay there -- the arm mesh itself
            # already renders the real joint angles, so this only drops the
            # supplementary colour-coded indicator, not the twin's core view.
            scene = getattr(viewer, "user_scn", None)
            if scene is not None:
                # Clear user scene by resetting the geom counter.
                scene.ngeom = 0
                for jname, adr in _link_qadr.items():
                    # Place a small sphere on each link body, colour-coded by joint angle.
                    ji = int(jname.rsplit("joint", 1)[1]) - 1  # 0..6
                    body_name = LINK_BODIES[ji]
                    bid = _link_body_id.get(body_name)
                    if bid is None or bid < 0:
                        continue
                    pos = data.xpos[bid].copy()
                    angle_deg = np.degrees(data.qpos[adr])
                    # Colour: green→0°, yellow→45°, red→≥90° (absolute deviation).
                    t = min(abs(angle_deg) / 90.0, 1.0)
                    rgba = [t, 1.0 - t, 0.0, 0.9]
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.01, 0, 0],
                        pos=pos,
                        mat=np.eye(3, dtype=np.float64).ravel(),
                        rgba=np.array(rgba, dtype=np.float32),
                    )
                    scene.ngeom += 1

            viewer.sync()

            now = time.time()
            if now - last_status >= 1.0:
                last_status = now
                _print_status(now, js, pose_src, _last_pose, pose_src_label)
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
