from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack

from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import FakeRobotBackend, FakeWorld, WorldPoseSource
from mile_franka.envs.franka_env import FrankaEnv

FAKE_ENV_ID = "Franka-Stack-Fake-v0"
SIM_ENV_ID = "Franka-Stack-Sim-v0"


def _build_fake_env(config: Optional[StackTaskConfig] = None) -> FrankaEnv:
    config = config if config is not None else StackTaskConfig()
    world = FakeWorld(config)
    return FrankaEnv(FakeRobotBackend(world), WorldPoseSource(world), config, mode="sim")


def _build_sim_env(config: Optional[StackTaskConfig] = None) -> FrankaEnv:
    """Multipanda MuJoCo-sim env. Imports ROS lazily (only when this id is made)."""
    from mile_franka.envs.ros_backend import MultipandaRosBackend
    from mile_franka.pose.mujoco_gt import MujocoGtPoseSource

    # Align with the MJCF stacking scene: 5 cm cubes (half-extent 0.025), floor at world z=0.
    # 300-step horizon: the scripted policy's 7-phase sequence needs ~150-250 steps against the
    # real impedance controller (which tracks more slowly than the fake backend).
    config = config if config is not None else StackTaskConfig(
        cube_size=0.05,
        table_z=0.0,
        workspace_low=np.array([0.40, -0.18, 0.02], dtype=np.float32),
        workspace_high=np.array([0.75, 0.18, 0.40], dtype=np.float32),
        max_steps=10_000,
    )
    backend = MultipandaRosBackend(
        config,
        apply_sim_gains=True,
        reset_controller_target_on_reset=True,
        # Home purely via cartesian prime->activate (the proven franka_sim_grasp_demo.py path):
        # cycling the controller inactive->active captures the current EE as the desired pose
        # (no dive), then _home() drives to the fixed home. The joint-space
        # move_to_start_example_controller is skipped -- its launch-time spawner loses the
        # cold-boot race and its activation was the source of the reset timeout.
        move_to_start_on_reset=False,
        env_step_period_s=0.1,
        home_steps=50,
        home_settle_s=0.8,
    )
    return FrankaEnv(backend, MujocoGtPoseSource(backend), config, mode="sim")


def make_franka_env(env: FrankaEnv, frame_stack: int = 4) -> gym.Env:
    """Apply FrameStack + FlattenObservation to a FrankaEnv (matches the MetaWorld path)."""
    return FlattenObservation(FrameStack(env, frame_stack))


def register_franka_envs() -> None:
    """Register Franka gym ids (idempotent)."""
    if FAKE_ENV_ID not in gym.registry:
        gym.register(id=FAKE_ENV_ID, entry_point=_build_fake_env)
    if SIM_ENV_ID not in gym.registry:
        gym.register(id=SIM_ENV_ID, entry_point=_build_sim_env)
