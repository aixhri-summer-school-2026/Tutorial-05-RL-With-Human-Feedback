"""Collect scripted-policy rollouts as imitation Transitions (spec sec 5.5).

Runs ScriptedStackPolicy on the wrapped (72,) FrankaEnv and records every (obs, action,
next_obs, done) step. The policy reads the current frame obs[-18:].
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from imitation.data.types import Transitions

from mile_franka.policies.scripted import ScriptedStackPolicy


def collect_scripted_demos(env: gym.Env, policy: ScriptedStackPolicy,
                           n_episodes: int, rng: np.random.Generator) -> Transitions:
    """Roll `policy` on wrapped `env` for n_episodes; return imitation Transitions."""
    obs_list, act_list, next_list, done_list = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        policy.reset(rng)
        for _ in range(env.unwrapped.config.max_steps):
            action = policy.act(np.asarray(obs)[-18:])
            next_obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(np.asarray(action, dtype=np.float32))
            next_list.append(np.asarray(next_obs, dtype=np.float32))
            done_list.append(done)
            obs = next_obs
            if done or info["success"]:
                break
    n = len(obs_list)
    return Transitions(
        obs=np.array(obs_list, dtype=np.float32),
        acts=np.array(act_list, dtype=np.float32),
        infos=np.array([{}] * n),
        next_obs=np.array(next_list, dtype=np.float32),
        dones=np.array(done_list, dtype=bool),
    )
