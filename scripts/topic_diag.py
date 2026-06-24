#!/usr/bin/env python3
"""Minimal topic diagnostic: subscribe to the three calibrate_camera topics
and print a counter + latest value every second.  Run in-container:

    python3 scripts/topic_diag.py
    python3 scripts/topic_diag.py --ee-topic /some/other/topic
"""
import argparse
import time

import numpy as np


def _ee_from_msg(msg) -> np.ndarray:
    p = msg.pose.position
    q = msg.pose.orientation
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--ee-topic", default="/cartesian_impedance/cartesian_pos_curr")
    ap.add_argument("--joint-states-topic", default="/joint_states")
    args = ap.parse_args()

    import rclpy
    from sensor_msgs.msg import Image as ImageMsg
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = rclpy.create_node("topic_diag")

    counters = {"image": 0, "ee": 0, "js": 0}
    latest = {"image": None, "ee": None, "js": None}

    def _image_cb(msg):
        counters["image"] += 1
        latest["image"] = f"{msg.width}x{msg.height}"

    def _ee_cb(msg):
        counters["ee"] += 1
        T = _ee_from_msg(msg)
        latest["ee"] = f"[{T[0,3]:+.3f}, {T[1,3]:+.3f}, {T[2,3]:+.3f}]"

    def _js_cb(msg):
        counters["js"] += 1
        arm = []
        for name, pos in zip(msg.name, msg.position):
            low = name.lower()
            if "finger" in low or "gripper" in low:
                continue
            arm.append((name, pos))
        arm.sort(key=lambda kv: kv[0])
        latest["js"] = f"{len(arm)} arm joints: {[n for n,_ in arm[:7]]}"

    node.create_subscription(ImageMsg, args.image_topic, _image_cb, 10)
    node.create_subscription(PoseStamped, args.ee_topic, _ee_cb, 10)
    node.create_subscription(JointState, args.joint_states_topic, _js_cb, 10)

    print(f"Listening — move the robot and watch counters:")
    print(f"  image:          {args.image_topic}")
    print(f"  ee:             {args.ee_topic}")
    print(f"  joint_states:   {args.joint_states_topic}")
    print()

    t0 = time.time()
    last_print = t0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - t0
                print(f"[{elapsed:4.0f}s] "
                      f"image={counters['image']:4d} ({latest['image'] or 'n/a':>12s})  "
                      f"ee={counters['ee']:4d} ({latest['ee'] or 'n/a':>30s})  "
                      f"js={counters['js']:4d} ({latest['js'] or 'n/a'})",
                      flush=True)
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
