#!/usr/bin/env python3
"""Run the BC base policy in Franka-Stack-Sim-v0 (no intervention) and report success rate.

Operator-initiated sim measurement to confirm the base policy is mediocre (spec 5.4).
Requires a live sim (`make sim-up` or `make sim-gui`).
"""
import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

from mile_franka.envs.registration import register_franka_envs, make_franka_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="trained_models/franka/base_policy")
    ap.add_argument("--env_name", default="Franka-Stack-Sim-v0")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    register_franka_envs()
    env = make_franka_env(gym.make(args.env_name))
    policy = ActorCriticPolicy.load(args.policy)
    policy.eval()

    successes, steps_used = [], []
    max_t = env.unwrapped.config.max_steps
    for ep in range(args.episodes):
        state, _ = env.reset()
        success = 0
        for t in range(max_t):
            action, _ = policy.predict(np.asarray(state), deterministic=True)
            state, _, terminated, truncated, info = env.step(action)
            if info["success"] or terminated or truncated:
                success = int(info["success"])
                steps_used.append(t + 1)
                break
        else:
            steps_used.append(max_t)
        successes.append(success)
        print(f"episode {ep}: success={success} steps={steps_used[-1]}")

    rate = float(np.mean(successes))
    print(f"\nBASE POLICY SIM SUCCESS RATE = {rate:.2f} "
          f"({sum(successes)}/{args.episodes}); mean steps={np.mean(steps_used):.0f}")
    env.close()


if __name__ == "__main__":
    main()
