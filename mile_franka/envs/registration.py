from __future__ import annotations

from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack

from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import FakeRobotBackend, FakeWorld, WorldPoseSource
from mile_franka.envs.franka_env import FrankaEnv

FAKE_ENV_ID = "Franka-Stack-Fake-v0"


def _build_fake_env(config: Optional[StackTaskConfig] = None) -> FrankaEnv:
    config = config if config is not None else StackTaskConfig()
    world = FakeWorld(config)
    return FrankaEnv(FakeRobotBackend(world), WorldPoseSource(world), config, mode="sim")


def make_franka_env(env: FrankaEnv, frame_stack: int = 4) -> gym.Env:
    """Apply FrameStack + FlattenObservation to a FrankaEnv (matches the MetaWorld path)."""
    return FlattenObservation(FrameStack(env, frame_stack))


def register_franka_envs() -> None:
    """Register Franka gym ids (idempotent)."""
    if FAKE_ENV_ID in gym.registry:
        return
    gym.register(id=FAKE_ENV_ID, entry_point=_build_fake_env)
