#!/usr/bin/env python3
"""Run the BC base policy in Franka-Stack-Sim-v0 (no intervention) and report success rate.

Operator-initiated sim measurement to confirm the base policy is mediocre (spec 5.4).
Requires a live sim (`make sim-up` or `make sim-gui`).
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np

from mile_franka.envs.registration import register_franka_envs, make_franka_env
from mile_franka.policies.bc import load_actor_critic_policy
from video_utils import start_recorder, stop_recorder, detect_crop as _detect_crop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="trained_models/franka/base_policy")
    ap.add_argument("--env_name", default="Franka-Stack-Sim-v0")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--video_dir", default=None,
                    help="if set, record one clip per episode here (episode_<i>.mp4)")
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":99"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--video_start_hold", type=float, default=0.5,
                    help="seconds to hold the reset scene before policy actions (recording on)")
    ap.add_argument("--video_end_hold", type=float, default=0.5,
                    help="seconds to hold the final scene before stopping the recorder")
    args = ap.parse_args()

    register_franka_envs()
    env = make_franka_env(gym.make(args.env_name))
    policy = load_actor_critic_policy(args.policy)
    policy.eval()

    successes, steps_used = [], []
    max_t = env.unwrapped.config.max_steps

    crop_filter = None
    if args.video_dir:
        print("detecting MuJoCo window crop region...")
        crop_filter = _detect_crop(args.display, args.size, args.fps)
        if crop_filter:
            print(f"  crop filter: {crop_filter}")

    for ep in range(args.episodes):
        state, _ = env.reset()
        # Start the recorder after reset so the first frame is the episode's start state.
        rec = None
        if args.video_dir:
            rec = start_recorder(os.path.join(args.video_dir, f"episode_{ep}.mp4"),
                                 args.display, args.size, args.fps, crop=crop_filter)
            time.sleep(args.video_start_hold)
        success = 0
        try:
            for t in range(max_t):
                # Sample, don't take the mean: a mean action that collapses to ~0 at
                # OOD/grasp states produces no motion -> identical obs -> the same ~0
                # action forever (deterministic closed-loop deadlock). See build_base_policy.
                action, _ = policy.predict(np.asarray(state), deterministic=False)
                state, _, terminated, truncated, info = env.step(action)
                if info["success"] or terminated or truncated:
                    success = int(info["success"])
                    steps_used.append(t + 1)
                    break
            else:
                steps_used.append(max_t)
        finally:
            if rec is not None:
                time.sleep(args.video_end_hold)
                stop_recorder(rec)
        successes.append(success)
        print(f"episode {ep}: success={success} steps={steps_used[-1]}")

    rate = float(np.mean(successes))
    print(f"\nBASE POLICY SIM SUCCESS RATE = {rate:.2f} "
          f"({sum(successes)}/{args.episodes}); mean steps={np.mean(steps_used):.0f}")
    env.close()


if __name__ == "__main__":
    main()
