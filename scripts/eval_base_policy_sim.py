#!/usr/bin/env python3
"""Run the BC base policy in Franka-Stack-Sim-v0 (no intervention) and report success rate.

Operator-initiated sim measurement to confirm the base policy is mediocre (spec 5.4).
Requires a live sim (`make sim-up` or `make sim-gui`).
"""
import argparse
import functools
import os
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from mile_franka.envs.registration import register_franka_envs, make_franka_env


def start_recorder(out_path, display, size, fps, crop=None):
    """ffmpeg x11grab recorder for the headless Xvfb display (mirrors franka_sim_rollout_record)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "x11grab", "-video_size", size,
           "-framerate", str(fps), "-i", display]
    if crop:
        cmd += ["-vf", crop]
    cmd += ["-pix_fmt", "yuv420p", out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_recorder(proc):
    try:
        proc.communicate(input=b"q", timeout=10)
    except Exception:
        proc.terminate()
        proc.wait(timeout=10)


def _detect_crop(display, size, fps):
    """Capture a few seconds, run cropdetect, return crop=W:H:X:Y or None."""
    import re
    import tempfile
    cmd = ["ffmpeg", "-y", "-f", "x11grab", "-video_size", size,
           "-framerate", str(fps), "-i", display,
           "-t", "2", "-vf", "cropdetect",
           "-pix_fmt", "yuv420p", "-f", "null", "/dev/null"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    matches = re.findall(r"crop=(\d+:\d+:\d+:\d+)", proc.stderr)
    if matches:
        return f"crop={matches[-1]}"
    return None


def _load_policy(path: str) -> ActorCriticPolicy:
    """Load an SB3 policy saved by build_base_policy.

    PyTorch>=2.6 flipped torch.load's `weights_only` default to True and refuses to unpickle
    the gymnasium Box stored in the SB3 checkpoint. SB3 2.3.x's load() doesn't expose the arg,
    so force weights_only=False for this call -- we created the checkpoint, so it's trusted.
    """
    orig_load = th.load
    th.load = functools.partial(orig_load, weights_only=False)
    try:
        return ActorCriticPolicy.load(path)
    finally:
        th.load = orig_load


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
    policy = _load_policy(args.policy)
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
