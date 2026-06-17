"""Run the scripted/base policy on the multipanda MuJoCo sim, record video + collect data.

Runs INSIDE the multipanda container against an already-launched sim
(franka_sim_stacking.launch.py with the stacking cubes). Records one video per
trajectory via ffmpeg x11grab and saves per-step (obs, action) transitions.

By default (--require_success true, --mediocre false) only successful episodes are
kept; retries up to --max_attempts times to collect --episodes successes.

Example (in container):
    DISPLAY=:99 python scripts/franka_sim_rollout_record.py \\
        --episodes 3 --out_dir output_dir/franka/rollouts/ \\
        --data output_dir/franka/sim_demos.npz
"""
import argparse
import os
import subprocess
import time

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack

from mile_franka.envs.registration import SIM_ENV_ID, register_franka_envs
from mile_franka.policies.scripted import ScriptedStackPolicy


def start_recorder(out_path, display, size, fps):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "x11grab", "-video_size", size,
           "-framerate", str(fps), "-i", display,
           "-pix_fmt", "yuv420p", out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_recorder(proc):
    try:
        proc.communicate(input=b"q", timeout=10)
    except Exception:
        proc.terminate()
        proc.wait(timeout=10)


def current_state(obs):
    frame = np.asarray(obs)[-18:]
    return {
        "ee": frame[:3].copy(),
        "width": float(frame[3]),
        "top": frame[4:7].copy(),
        "bottom": frame[11:14].copy(),
    }


def format_state(state):
    return (
        f"ee={np.array2string(state['ee'], precision=3)} "
        f"width={state['width']:.3f} "
        f"top={np.array2string(state['top'], precision=3)} "
        f"bottom={np.array2string(state['bottom'], precision=3)}"
    )


def reset_episode(env, policy, rng, seed):
    obs, _ = env.reset(seed=seed)
    policy.reset(rng)
    return obs


def run_episode(env, policy, obs):
    obs_list, act_list = [], []
    info = {}
    for _ in range(env.unwrapped.config.max_steps):
        action = policy.act(np.asarray(obs)[-18:])
        next_obs, _, terminated, truncated, info = env.step(action)
        obs_list.append(np.asarray(obs, dtype=np.float32))
        act_list.append(np.asarray(action, dtype=np.float32))
        obs = next_obs
        if info.get("success") or terminated or truncated:
            break
    return obs_list, act_list, bool(info.get("success", False)), obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3,
                    help="number of (successful) episodes to collect")
    ap.add_argument("--out", default=None,
                    help="(legacy) path for a single combined video")
    ap.add_argument("--out_dir", default=None,
                    help="directory for per-episode files: ep0.mp4, ep1.mp4, …")
    ap.add_argument("--data", default="/tmp/demos.npz")
    ap.add_argument("--mediocre", type=lambda s: s.lower() != "false", default=False,
                    help="use mediocre (noisy) policy; default False = expert")
    ap.add_argument("--require_success", type=lambda s: s.lower() != "false", default=True,
                    help="discard and retry failed episodes (default True)")
    ap.add_argument("--max_attempts", type=int, default=0,
                    help="max attempts before giving up (0 = 3 × --episodes)")
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":99"))
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_start_hold", type=float, default=0.5,
                    help="seconds to hold the reset scene before policy actions")
    ap.add_argument("--video_end_hold", type=float, default=0.5,
                    help="seconds to keep recording after the episode ends")
    args = ap.parse_args()

    max_attempts = args.max_attempts if args.max_attempts > 0 else 3 * args.episodes

    register_franka_envs()
    env = FlattenObservation(FrameStack(gym.make(SIM_ENV_ID), 4))
    rng = np.random.default_rng(args.seed)
    policy = ScriptedStackPolicy(getattr(env.unwrapped, "config", None),
                                 mediocre=args.mediocre)

    # Legacy: single combined video for all episodes. Prefer --out_dir: per-episode
    # videos start after reset, so their first frame is the actual episode start state.
    combined_rec = None

    all_obs, all_acts, all_ep = [], [], []
    saved = 0
    successes = 0
    attempt = 0

    try:
        while saved < args.episodes and attempt < max_attempts:
            ep_path = None
            ep_rec = None
            obs = reset_episode(env, policy, rng, args.seed + attempt)
            start_state = current_state(obs)
            print(f"attempt {attempt}: start {format_state(start_state)}")

            if args.out and not args.out_dir and combined_rec is None:
                combined_rec = start_recorder(args.out, args.display, args.size, args.fps)
                time.sleep(args.video_start_hold)

            if args.out_dir:
                ep_path = os.path.join(args.out_dir, f"ep{saved}.mp4")
                os.makedirs(args.out_dir, exist_ok=True)
                ep_rec = start_recorder(ep_path, args.display, args.size, args.fps)
                time.sleep(args.video_start_hold)

            obs_list, act_list, success, final_obs = run_episode(env, policy, obs)
            end_state = current_state(final_obs)
            if ep_rec is not None or combined_rec is not None:
                time.sleep(args.video_end_hold)

            if ep_rec is not None:
                stop_recorder(ep_rec)

            print(
                f"attempt {attempt}: end success={success} steps={len(obs_list)} "
                f"{format_state(end_state)}"
            )

            if success or not args.require_success:
                if success:
                    successes += 1
                for o, a in zip(obs_list, act_list):
                    all_obs.append(o)
                    all_acts.append(a)
                    all_ep.append(saved)
                print(f"  -> saved as episode {saved}")
                saved += 1
            else:
                if ep_path and os.path.exists(ep_path):
                    os.unlink(ep_path)
                print(f"  -> discarded (failed)")

            attempt += 1

    finally:
        if combined_rec:
            stop_recorder(combined_rec)
        env.close()

    if saved < args.episodes:
        print(f"WARNING: only {saved}/{args.episodes} successful episodes "
              f"after {attempt} attempts")

    np.savez_compressed(args.data,
                        obs=np.array(all_obs, dtype=np.float32),
                        acts=np.array(all_acts, dtype=np.float32),
                        episode=np.array(all_ep, dtype=np.int64))
    rate = successes / max(attempt, 1)
    print(f"success rate = {rate:.2f} ({successes} success / {attempt} attempts)")
    print(f"saved {len(all_obs)} transitions -> {args.data}")
    if args.out_dir:
        print(f"saved {saved} episode videos -> {args.out_dir}")
    elif args.out:
        print(f"saved video -> {args.out}")


if __name__ == "__main__":
    main()
