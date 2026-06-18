"""Build the mediocre BC base policy for MILE-on-Franka (spec sec 5.5, gate 3).

Collects scripted (mediocre) demos on the fake FrankaEnv, BC-distills an ActorCriticPolicy,
saves it, and reports its rollout success rate.
"""
import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack
from imitation.data.types import Transitions

from mile_franka.envs.registration import FAKE_ENV_ID, SIM_ENV_ID, register_franka_envs
from mile_franka.policies.bc import train_bc
from mile_franka.policies.demos import collect_scripted_demos
from mile_franka.policies.scripted import ScriptedStackPolicy


def eval_policy(env, policy, n_episodes, rng):
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(env.unwrapped.config.max_steps):
            action, _ = policy.predict(np.asarray(obs), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if info["success"]:
                successes += 1
                break
            if terminated or truncated:
                break
    return successes / n_episodes


def load_npz_transitions(path):
    d = np.load(path, allow_pickle=False)
    obs = d["obs"].astype(np.float32)
    acts = d["acts"].astype(np.float32)
    ep = d["episode"]
    n = len(obs)
    next_obs = np.empty_like(obs)
    dones = np.zeros(n, dtype=bool)
    for i in range(n):
        if i + 1 < n and ep[i + 1] == ep[i]:
            next_obs[i] = obs[i + 1]
        else:
            next_obs[i] = obs[i]
            dones[i] = True
    return Transitions(obs=obs, acts=acts, infos=np.array([{}] * n),
                       next_obs=next_obs, dones=dones)


def report_bc_fit(policy, transitions):
    obs = transitions.obs.astype(np.float32)
    acts = transitions.acts.astype(np.float32)
    preds = []
    for o in obs:
        action, _ = policy.predict(o, deterministic=True)
        preds.append(action)
    preds = np.asarray(preds, dtype=np.float32)
    mae = np.mean(np.abs(preds - acts), axis=0)
    grip_acc = float(np.mean((preds[:, 3] > 0) == (acts[:, 3] > 0)))
    print(
        "BC train fit: "
        f"mae={np.array2string(mae, precision=4)} "
        f"grip_sign_acc={grip_acc:.3f} "
        f"demo_close_frac={np.mean(acts[:, 3] > 0):.3f} "
        f"pred_close_frac={np.mean(preds[:, 3] > 0):.3f}"
    )

    frame = obs[:, -18:]
    ee = frame[:, :3]
    top = frame[:, 4:7]
    near_low = (
        (np.linalg.norm(ee[:, :2] - top[:, :2], axis=1) < 0.015)
        & (top[:, 2] < 0.05)
    )
    if np.any(near_low):
        near_acc = float(np.mean(
            (preds[near_low, 3] > 0) == (acts[near_low, 3] > 0)))
        print(
            "BC train fit near low cube: "
            f"n={int(np.sum(near_low))} "
            f"grip_sign_acc={near_acc:.3f} "
            f"demo_close_frac={np.mean(acts[near_low, 3] > 0):.3f} "
            f"pred_close_frac={np.mean(preds[near_low, 3] > 0):.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default=SIM_ENV_ID)
    ap.add_argument("--demos", default=None,
                    help="path to a sim-demo .npz; skips scripted collection when set")
    ap.add_argument("--demo_episodes", type=int, default=60)
    ap.add_argument("--bc_epochs", type=int, default=100)
    ap.add_argument("--bc_batch_size", type=int, default=256)
    ap.add_argument("--bc_lr", type=float, default=1e-3)
    ap.add_argument("--bc_ent_weight", type=float, default=0.0)
    ap.add_argument("--bc_l2_weight", type=float, default=0.0)
    ap.add_argument("--eval_episodes", type=int, default=30)
    ap.add_argument("--mediocre", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--save_path", default="trained_models/franka/base_policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    register_franka_envs()
    rng = np.random.default_rng(args.seed)

    if args.demos:
        # BC from a pre-collected .npz needs no robot. The fake and sim envs share
        # identical obs/action spaces, so read the spaces from the no-ROS fake env
        # and only touch the (live) --env_id env when an eval rollout is requested.
        print(f"Loading demos from {args.demos} ...")
        demos = load_npz_transitions(args.demos)
        space_env = FlattenObservation(FrameStack(gym.make(FAKE_ENV_ID), 4))
        obs_space, act_space = space_env.observation_space, space_env.action_space
    else:
        env = FlattenObservation(FrameStack(gym.make(args.env_id), 4))
        obs_space, act_space = env.observation_space, env.action_space
        scripted = ScriptedStackPolicy(getattr(env.unwrapped, "config", None),
                                       mediocre=args.mediocre)
        print(f"Collecting {args.demo_episodes} scripted demos (mediocre={args.mediocre})...")
        demos = collect_scripted_demos(env, scripted, args.demo_episodes, rng)
    print(
        f"Using {len(demos.obs)} transitions; BC training {args.bc_epochs} epochs "
        f"batch_size={args.bc_batch_size} lr={args.bc_lr} "
        f"ent_weight={args.bc_ent_weight} l2_weight={args.bc_l2_weight}..."
    )
    policy = train_bc(
        demos, obs_space, act_space, rng,
        n_epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        lr=args.bc_lr,
        ent_weight=args.bc_ent_weight,
        l2_weight=args.bc_l2_weight,
    )
    report_bc_fit(policy, demos)

    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    policy.save(args.save_path)
    if args.eval_episodes > 0:
        # eval rolls out against the live --env_id env (the sim must be up).
        eval_env = FlattenObservation(FrameStack(gym.make(args.env_id), 4))
        rate = eval_policy(eval_env, policy, args.eval_episodes, rng)
        print(f"Saved base policy to {args.save_path}; success rate = {rate:.2f}")
    else:
        print(f"Saved base policy to {args.save_path}; success rate = 0.00")


if __name__ == "__main__":
    main()
