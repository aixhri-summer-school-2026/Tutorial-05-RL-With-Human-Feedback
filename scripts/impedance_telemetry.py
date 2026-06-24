#!/usr/bin/env python3
"""Live telemetry recorder for the FR3 Cartesian impedance bringup.

Subscribes to the controller's own debug topics (published at pub_frequency when the
controller is ACTIVE) and prints, at a throttled rate, the quantities that matter for
diagnosing a reflex WITHOUT guessing:

  - tracking error  |curr - des_filt|   (m)
  - max |dq| and WHICH joint            (rad/s)  <- the joint_velocity_violation signal
  - max |tau| and which joint           (Nm)

It also remembers and prints the worst (max) |dq| seen so far, so after a reflex you can
see exactly which joint spiked and how high. No hardware writes; safe to run alongside
the smoke test. Run inside a container with rclpy on the same DDS graph.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_STATE_TOPIC = "/cartesian_impedance/joint_state"      # q, dq, tau (effort=tau_J_d)
DES_TOPIC = "/cartesian_impedance/cartesian_pos_des_filt"   # filtered desired pose
CURR_TOPIC = "/cartesian_impedance/cartesian_pos_curr"      # current EE pose


class Telemetry(Node):
    def __init__(self, hz: float, dq_warn: float):
        super().__init__("impedance_telemetry")
        self.dq_warn = dq_warn
        self.curr = None
        self.des = None
        self.worst_dq = 0.0
        self.worst_dq_joint = -1
        self.last_print = 0.0
        self.period = 1.0 / hz
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._js, 10)
        self.create_subscription(PoseStamped, CURR_TOPIC, self._curr, 10)
        self.create_subscription(PoseStamped, DES_TOPIC, self._des, 10)

    def _curr(self, m):
        self.curr = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])

    def _des(self, m):
        self.des = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])

    def _js(self, m):
        dq = np.asarray(m.velocity, dtype=float)
        tau = np.asarray(m.effort, dtype=float)
        if dq.size == 0:
            return
        j = int(np.argmax(np.abs(dq)))
        maxdq = float(abs(dq[j]))
        if maxdq > self.worst_dq:
            self.worst_dq, self.worst_dq_joint = maxdq, j
        now = time.time()
        # Always print immediately on a velocity spike; otherwise throttle.
        spike = maxdq > self.dq_warn
        if not spike and (now - self.last_print) < self.period:
            return
        self.last_print = now
        err = (float(np.linalg.norm(self.curr - self.des))
               if self.curr is not None and self.des is not None else float("nan"))
        tj = int(np.argmax(np.abs(tau))) if tau.size else -1
        maxtau = float(abs(tau[tj])) if tau.size else float("nan")
        flag = "  <<< DQ SPIKE" if spike else ""
        print(f"err={err*100:6.2f}cm  maxdq=j{j+1} {maxdq:5.2f} rad/s  "
              f"maxtau=j{tj+1} {maxtau:6.2f} Nm  (worst dq so far: j{self.worst_dq_joint+1} "
              f"{self.worst_dq:.2f}){flag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=5.0, help="throttled print rate")
    ap.add_argument("--dq-warn", type=float, default=0.5,
                    help="print immediately when any |dq| exceeds this (rad/s)")
    args = ap.parse_args()
    rclpy.init()
    node = Telemetry(args.hz, args.dq_warn)
    print(f"[telemetry] listening on {JOINT_STATE_TOPIC} (controller must be ACTIVE)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[telemetry] worst |dq| seen: joint {node.worst_dq_joint+1} "
              f"= {node.worst_dq:.3f} rad/s")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
