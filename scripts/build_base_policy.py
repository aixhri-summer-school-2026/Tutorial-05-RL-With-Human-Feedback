"""Build the mediocre BC base policy for MILE-on-Franka (spec sec 5.5, gate 3).

Collects scripted (mediocre) demos on the fake FrankaEnv, BC-distills an ActorCriticPolicy,
saves it, and reports its rollout success rate.
"""
import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack

from mile_franka.envs.registration import FAKE_ENV_ID, register_franka_envs
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default=FAKE_ENV_ID)
    ap.add_argument("--demo_episodes", type=int, default=60)
    ap.add_argument("--bc_epochs", type=int, default=30)
    ap.add_argument("--eval_episodes", type=int, default=30)
    ap.add_argument("--mediocre", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--save_path", default="trained_models/franka/base_policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    register_franka_envs()
    env = FlattenObservation(FrameStack(gym.make(args.env_id), 4))
    rng = np.random.default_rng(args.seed)

    scripted = ScriptedStackPolicy(mediocre=args.mediocre)
    print(f"Collecting {args.demo_episodes} scripted demos (mediocre={args.mediocre})...")
    demos = collect_scripted_demos(env, scripted, args.demo_episodes, rng)
    print(f"Collected {len(demos.obs)} transitions; BC training {args.bc_epochs} epochs...")
    policy = train_bc(demos, env.observation_space, env.action_space, rng,
                      n_epochs=args.bc_epochs)

    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    policy.save(args.save_path)
    rate = eval_policy(env, policy, args.eval_episodes, rng)
    print(f"Saved base policy to {args.save_path}; success rate = {rate:.2f}")


if __name__ == "__main__":
    main()
