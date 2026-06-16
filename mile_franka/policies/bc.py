"""Build and BC-train the MILE-compatible base policy (spec sec 5.5).

build_actor_critic_policy reproduces the arch train_mile.py uses for bc policies/mental
models so the saved policy round-trips through ActorCriticPolicy.load. train_bc distills
scripted demos into it with imitation's BC, yielding the differentiable Gaussian rollout
policy MILE requires.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.util.networks import RunningNorm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn


def build_actor_critic_policy(observation_space: gym.spaces.Space,
                              action_space: gym.spaces.Space,
                              device: str = "cpu") -> ActorCriticPolicy:
    """ActorCriticPolicy matching train_mile.py's bc arch (net [256,256] + RunningNorm)."""
    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    policy = ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=get_schedule_fn(1),
        net_arch=[256, 256],
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    )
    policy.to(resolved)
    return policy


def train_bc(transitions: Transitions, observation_space: gym.spaces.Space,
             action_space: gym.spaces.Space, rng: np.random.Generator,
             n_epochs: int = 20, batch_size: int = 64,
             device: str = "cpu") -> ActorCriticPolicy:
    """Distill demonstrations into a fresh MILE-compatible ActorCriticPolicy."""
    policy = build_actor_critic_policy(observation_space, action_space, device=device)
    bc = BC(
        observation_space=observation_space,
        action_space=action_space,
        rng=rng,
        policy=policy,
        demonstrations=transitions,
        batch_size=batch_size,
        device=device,
    )
    bc.train(n_epochs=n_epochs, progress_bar=False)
    return bc.policy
