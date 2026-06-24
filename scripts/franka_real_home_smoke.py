#!/usr/bin/env python3
"""Operator-gated real Franka home + small Cartesian motion smoke test.

This deliberately avoids policy execution and object pose sources. It only verifies the
real controller path needed before eval-real:

1. activate custom_cartesian_impedance_controller (Effort) via move_to_start → Cartesian home
2. command the corrected wrist-down Cartesian home
3. trace a small square around home

Run only with an operator at the robot and the physical stop reachable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from mile_franka.config import FR3_DOWN_QUAT, MULTIPANDA_DOWN_QUAT, StackTaskConfig
from mile_franka.envs.ros_backend import MultipandaRosBackend


def _confirm(prompt: str) -> None:
    print()
    print(f"  >>> {prompt}")
    print("  >>> Press Enter to continue, Ctrl-C to abort.")
    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted by operator.")
        sys.exit(0)


def _ramp_cartesian(backend: MultipandaRosBackend, target: np.ndarray, duration_s: float) -> None:
    # set_equilibrium_pose advances at most setpoint_substep_m toward the target per call
    # (one tick, so the controller's smoothstep finishes before the next setpoint). Step
    # toward the target repeatedly until it is reached or the time budget runs out.
    deadline = time.time() + max(duration_s, 1.0)
    while time.time() < deadline:
        backend.set_equilibrium_pose(target, backend.down_quat)
        if float(np.linalg.norm(backend.get_ee_position() - target)) < 0.01:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Home the real Franka and trace a tiny square.")
    parser.add_argument("--step", type=float, default=0.1,
                        help="square side length in meters; keep small for first motion")
    parser.add_argument("--segment-s", type=float, default=4.0,
                        help="duration for each Cartesian segment")
    parser.add_argument("--hold-s", type=float, default=0.8,
                        help="hold time after each segment")
    parser.add_argument("--home-steps", type=int, default=90,
                        help="interpolation steps used by backend home ramp")
    parser.add_argument("--home-timeout-s", type=float, default=10.0,
                        help="maximum Cartesian home settle time")
    parser.add_argument("--move-to-start-hold-s", type=float, default=8.0,
                        help="time to let move_to_start run before Cartesian activation")
    parser.add_argument("--no-square", action="store_true",
                        help="only home; do not trace the small square")
    args = parser.parse_args()

    if args.step <= 0.0 or args.step > 0.1:
        raise ValueError("--step must be in (0, 0.1] for this smoke test")

    real_stack = os.environ.get("MILE_REAL_STACK", "multipanda").lower()
    use_fr3_pose = real_stack in ("fr3", "franka_ros2", "fr3_pose")
    apply_sim_gains = os.environ.get("MILE_APPLY_SIM_GAINS", "0").lower() in (
        "1", "true", "yes")

    config = StackTaskConfig(
        cube_size=0.05,
        table_z=0.0,
        workspace_low=np.array([0.35, -0.25, 0.02], dtype=np.float32),
        workspace_high=np.array([0.75, 0.25, 0.35], dtype=np.float32),
        max_steps=10_000,
    )

    print("Real Franka home smoke test")
    print(f"  step             : {args.step:.3f} m")
    print(f"  segment duration : {args.segment_s:.1f} s")
    print(f"  real stack       : {real_stack}")
    print(f"  apply sim gains  : {apply_sim_gains}")
    _confirm("ARM WILL MOVE. Clear the workspace and keep the e-stop/stop button ready.")

    backend = MultipandaRosBackend(
        config,
        sim=False,
        base_frame=("fr3_link0" if use_fr3_pose else "panda_link0"),
        randomize_on_reset=False,
        grasp_action="/franka_gripper_node/grasp",
        controller_name=os.environ.get(
            "MILE_CONTROLLER",
            "custom_cartesian_impedance_controller"),
        # The impedance controller activates at whatever orientation the arm is
        # currently at, then _home() ramps toward DOWN_QUAT under impedance control.
        # Don't use move_to_start: the Effort->CartesianPose->Effort mode round-trip
        # trips communication_constraints_violation on the FR3.
        move_to_start_on_reset=False,
        reset_controller_target_on_reset=True,
        apply_sim_gains=apply_sim_gains,
        env_step_period_s=0.1,
        move_to_start_hold_s=args.move_to_start_hold_s,
        home_steps=args.home_steps,
        home_settle_tol=0.015,
        home_settle_timeout_s=args.home_timeout_s,
        setpoint_substep_m=float(os.environ.get("MILE_SUBSTEP_M", "0.003")),
        down_quat=(FR3_DOWN_QUAT if use_fr3_pose else MULTIPANDA_DOWN_QUAT),
    )
    print(f"  setpoint_substep_m: {backend.setpoint_substep_m} m  "
          f"(~{1.875 * backend.setpoint_substep_m / 0.1 * 100:.1f} cm/s peak; "
          f"override with MILE_SUBSTEP_M=..)")

    try:
        print("\nHoming: reset active stack -> wrist-down Cartesian home")
        backend.reset(np.random.default_rng(0))
        home = backend.get_ee_position()
        print(f"Home EE position: {home.tolist()}")

        if args.no_square:
            print("Home-only smoke test complete.")
            return

        _confirm("About to trace a small square around home.")
        points = [
            home + np.array([args.step, 0.0, 0.0], dtype=np.float32),
            home + np.array([args.step, args.step, 0.0], dtype=np.float32),
            home + np.array([0.0, args.step, 0.0], dtype=np.float32),
            home,
        ]
        for index, target in enumerate(points, start=1):
            print(f"Segment {index}/{len(points)} target: {target.tolist()}")
            _ramp_cartesian(backend, target, args.segment_s)
            time.sleep(args.hold_s)
            current = backend.get_ee_position()
            err = float(np.linalg.norm(current - target))
            print(f"  current: {current.tolist()}  error: {err * 100.0:.1f} cm")

        print("Motion smoke test complete.")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
