#!/usr/bin/env python3
"""Command a target wrist orientation at the CURRENT position and report the result.

Used to test whether the FR3 can hold clean axis-aligned down (xyzw=[1,0,0,0],
RPY[180,0,0]) under the soft Cartesian-impedance controller, vs the -45deg-yawed
FR3_DOWN_QUAT. Holds position fixed; only the orientation target changes. Reports the
resulting EE quaternion and its yaw about vertical so the 45deg offset is quantified.

Run with the impedance controller ACTIVE and an operator at the e-stop.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

CURR = "/cartesian_impedance/cartesian_pos_curr"
EQ = "/cartesian_impedance/equilibrium_pose"


def quat_yaw_deg(q):  # xyzw -> yaw of the wrist about base z, after the 180 flip
    x, y, z, w = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Orient(Node):
    def __init__(self):
        super().__init__("impedance_orient_test")
        self.pos = None
        self.quat = None
        self.create_subscription(PoseStamped, CURR, self._curr, 10)
        self.pub = self.create_publisher(PoseStamped, EQ, 10)

    def _curr(self, m):
        p, o = m.pose.position, m.pose.orientation
        self.pos = np.array([p.x, p.y, p.z])
        self.quat = np.array([o.x, o.y, o.z, o.w])

    def wait(self, t=5.0):
        d = time.time() + t
        while self.pos is None and time.time() < d:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.pos is not None

    def publish(self, pos, quat):
        m = PoseStamped()
        m.header.frame_id = "fr3_link0"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
        (m.pose.orientation.x, m.pose.orientation.y,
         m.pose.orientation.z, m.pose.orientation.w) = map(float, quat)
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quat", type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0],
                    help="target xyzw (default clean down [1,0,0,0])")
    ap.add_argument("--settle", type=float, default=6.0)
    args = ap.parse_args()
    rclpy.init()
    n = Orient()
    if not n.wait():
        print("ERROR: no pose -- controller active?"); return
    pos = n.pos.copy()
    q0 = n.quat.copy()
    target = np.array(args.quat) / np.linalg.norm(args.quat)
    print(f"start  quat: {q0.round(4).tolist()}  yaw={quat_yaw_deg(q0):+.1f} deg")
    print(f"target quat: {target.round(4).tolist()}  yaw={quat_yaw_deg(target):+.1f} deg")
    print(f"holding position {pos.round(4).tolist()} fixed, rotating wrist...")
    d = time.time() + args.settle
    while time.time() < d:
        n.publish(pos, target)
        rclpy.spin_once(n, timeout_sec=0.05)
        time.sleep(0.05)
    qf = n.quat.copy()
    print(f"final  quat: {qf.round(4).tolist()}  yaw={quat_yaw_deg(qf):+.1f} deg")
    print(f"reached yaw within {abs(quat_yaw_deg(qf) - quat_yaw_deg(target)):.1f} deg of target; "
          f"position drift {np.linalg.norm(n.pos - pos) * 100:.2f} cm")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
