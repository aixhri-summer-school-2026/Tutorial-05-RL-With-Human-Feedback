#!/usr/bin/env python3
"""Run a policy on the real FR3 (Franka-Stack-Real-v0) and report success rate.

**Hardware prerequisites:**
- hucebot multipanda_ros2 controller docker running on the FR3 control PC
- Our container on the same DDS graph (``make up`` with ``network_mode: host``)
- ``make apriltag-up`` running (RealSense + AprilTag detection + calibration static tf)
- Valid ``config/camera_calib.yaml`` from the calibration capture script
- Operator present — this script moves the real arm on every reset

**Safety:** ``auto_eval`` is always ``false`` on hardware. This script is **explicit
human-supervised eval**: the operator stands at the robot, watches every episode, and
hits the physical emergency stop if anything goes wrong. The arm moves to a known joint
home on each reset, then to a Cartesian home above the workspace; the policy only
commands small (≤5.5 cm) delta actions.

**Usage (in-container):**

    python3 scripts/eval_base_policy_real.py --episodes 10
    python3 scripts/eval_base_policy_real.py --policy trained_models/franka/policy --episodes 5
"""
import argparse
import sys
import time

import numpy as np

from mile_franka.envs.registration import register_franka_envs, make_franka_env
from mile_franka.policies.bc import load_actor_critic_policy


def _confirm(prompt: str) -> None:
    """Print a prompt and wait for the operator to press Enter (or Ctrl-C to abort)."""
    print()
    print(f"  >>> {prompt}")
    print("  >>> Press Enter to continue, Ctrl-C to abort.")
    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted by operator.")
        sys.exit(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a policy on the real FR3.")
    ap.add_argument("--policy", default="trained_models/franka/base_policy",
                    help="path to a saved SB3 ActorCriticPolicy")
    ap.add_argument("--env_name", default="Franka-Stack-Real-v0")
    ap.add_argument("--episodes", type=int, default=5,
                    help="number of evaluation episodes")
    args = ap.parse_args()

    register_franka_envs()
    import gymnasium as gym
    env = make_franka_env(gym.make(args.env_name))
    policy = load_actor_critic_policy(args.policy)
    policy.eval()

    print(f"Policy   : {args.policy}")
    print(f"Env      : {args.env_name}")
    print(f"Episodes : {args.episodes}")
    _confirm("ARM WILL MOVE on reset. Clear the workspace and stand clear.")

    successes, steps_used, intervened = [], [], []
    max_t = env.unwrapped.config.max_steps
    for ep in range(args.episodes):
        # --- Reset (moves the arm: joint home → activate Cartesian → Cartesian home) ---
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{args.episodes}")
        _confirm(f"Episode {ep+1}: about to RESET (arm will move to home).")

        state, _ = env.reset()
        unwrapped = env.unwrapped
        ee0 = np.asarray(unwrapped.backend.get_ee_position(), dtype=np.float32)
        print("Reset complete — arm is at Cartesian home.")
        print(f"  home EE   : [{ee0[0]:.3f} {ee0[1]:.3f} {ee0[2]:.3f}]")
        print(f"  down_quat : {np.round(getattr(unwrapped.backend, 'down_quat', None), 4).tolist()}")
        print(f"  action_scale: {unwrapped.config.action_scale} m/tick")

        # --- Place cubes ---
        _confirm("Place the cubes in the workspace, then press Enter.")
        print("Running policy — watch the arm and be ready on the e-stop.")

        # --- Run policy ---
        success = 0
        try:
            for t in range(max_t):
                # Sample, don't take the mean: a mean action that collapses to ~0 at
                # OOD/grasp states produces no motion -> identical obs -> the same ~0
                # action forever (deterministic closed-loop deadlock). See build_base_policy.
                action, _ = policy.predict(np.asarray(state), deterministic=False)
                ee_before = np.asarray(unwrapped.backend.get_ee_position(), dtype=np.float32)
                state, reward, terminated, truncated, info = env.step(action)
                ee_after = np.asarray(unwrapped.backend.get_ee_position(), dtype=np.float32)
                act = np.asarray(action, dtype=np.float32).reshape(4)
                cmd_cm = float(np.linalg.norm(act[:3]) * unwrapped.config.action_scale * 100.0)
                moved_cm = float(np.linalg.norm(ee_after - ee_before) * 100.0)
                tgt = np.round(unwrapped._ee_target, 3).tolist()
                print(f"  t={t:3d} act=[{act[0]:+.3f} {act[1]:+.3f} {act[2]:+.3f} g={act[3]:+.3f}] "
                      f"cmd={cmd_cm:4.1f}cm moved={moved_cm:4.2f}cm "
                      f"ee=[{ee_after[0]:.3f} {ee_after[1]:.3f} {ee_after[2]:.3f}] tgt={tgt} "
                      f"rew={reward:+.2f} succ={info.get('success')}")
                if info.get("success") or terminated or truncated:
                    success = int(info.get("success", 0))
                    steps_used.append(t + 1)
                    break
            else:
                steps_used.append(max_t)
        except KeyboardInterrupt:
            print("Episode interrupted by operator (Ctrl-C).")
            success = 0
            steps_used.append(0)

        successes.append(success)
        status = "SUCCESS" if success else "no stack"
        print(f"Episode {ep}: {status}  steps={steps_used[-1]}")

    # --- Summary ---
    rate = float(np.mean(successes))
    mean_steps = np.mean(steps_used) if steps_used else 0
    print(f"\n{'='*50}")
    print(f"REAL FR3 POLICY EVAL")
    print(f"  Success rate : {rate:.2f} ({sum(successes)}/{args.episodes})")
    print(f"  Mean steps   : {mean_steps:.0f}")
    print(f"  Policy       : {args.policy}")

    env.close()


if __name__ == "__main__":
    main()
