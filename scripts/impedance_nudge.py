#!/usr/bin/env python3
"""Minimal motion test for the FR3 Cartesian impedance controller.

Reads the current EE pose from /cartesian_impedance/cartesian_pos_curr, then commands a
single small equilibrium step (default +2 cm in x) and reports how close the arm gets.
The controller's internal filter ramps the desired pose, so one message produces a smooth
move. No scripted home, no frames juggling -- isolates "can it move?" from the smoke test.

Run with the controller ALREADY active and an operator at the e-stop.
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


class Nudge(Node):
    def __init__(self):
        super().__init__("impedance_nudge")
        self.curr = None
        self.quat = None
        self.create_subscription(PoseStamped, CURR, self._curr, 10)
        self.pub = self.create_publisher(PoseStamped, EQ, 10)

    def _curr(self, m):
        p, o = m.pose.position, m.pose.orientation
        self.curr = np.array([p.x, p.y, p.z])
        self.quat = np.array([o.x, o.y, o.z, o.w])

    def wait_pose(self, timeout=5.0):
        t = time.time()
        while self.curr is None and time.time() - t < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.curr is not None

    def publish(self, pos, quat):
        m = PoseStamped()
        m.header.frame_id = "fr3_link0"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, quat)
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    ap.add_argument("--settle", type=float, default=4.0)
    args = ap.parse_args()
    rclpy.init()
    n = Nudge()
    if not n.wait_pose():
        print("ERROR: no pose on cartesian_pos_curr -- is the controller active?")
        return
    start = n.curr.copy()
    quat = n.quat.copy()
    target = start + np.array([args.dx, args.dy, args.dz])
    print(f"start  EE: {start.round(4).tolist()}")
    print(f"target EE: {target.round(4).tolist()}  (delta {np.array([args.dx,args.dy,args.dz]).tolist()})")
    deadline = time.time() + args.settle
    while time.time() < deadline:
        n.publish(target, quat)
        rclpy.spin_once(n, timeout_sec=0.05)
        time.sleep(0.05)
    err = float(np.linalg.norm(n.curr - target))
    moved = float(np.linalg.norm(n.curr - start))
    print(f"final  EE: {n.curr.round(4).tolist()}")
    print(f"moved {moved*100:.2f} cm toward a {np.linalg.norm(target-start)*100:.2f} cm target; "
          f"remaining error {err*100:.2f} cm")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
