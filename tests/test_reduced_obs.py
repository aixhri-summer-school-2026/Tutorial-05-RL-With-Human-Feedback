"""Reduced-observation env + scripted randomization tests (no ROS/hardware needed)."""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack

from mile_franka.envs.registration import (
    FAKE_ENV_ID, FRANKA_FRAME_STACK, register_franka_envs)
from mile_franka.policies.scripted import ScriptedStackPolicy
from mile_franka.pose.canonical import TablePlaneCanonicalizer

register_franka_envs()


def test_default_frame_is_9_dim():
    env = gym.make(FAKE_ENV_ID)
    assert env.observation_space.shape == (9,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9,)


def test_include_bottom_z_is_10_dim():
    env = gym.make(FAKE_ENV_ID, obs_include_bottom_z=True)
    assert env.observation_space.shape == (10,)


def test_privileged_frame_is_full_18_dim():
    env = gym.make(FAKE_ENV_ID)
    env.reset(seed=0)
    assert env.unwrapped.privileged_frame().shape == (18,)


def test_framestack_depth_honored_end_to_end():
    env = FlattenObservation(FrameStack(gym.make(FAKE_ENV_ID), FRANKA_FRAME_STACK))
    assert env.observation_space.shape == (9 * FRANKA_FRAME_STACK,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9 * FRANKA_FRAME_STACK,)


def test_z_canonicalizer_shifts_ee_and_top_z():
    # Inject a canonicalizer into the env and confirm only the vertical channels move by delta.
    env = gym.make(FAKE_ENV_ID).unwrapped
    env.reset(seed=1)
    raw = env._build_obs().copy()              # canonicalizer is None by default
    bottom_z = float(env.pose_source.get_pose("bottom_cube").to_array()[2])
    top_z = float(env.pose_source.get_pose("top_cube").to_array()[2])
    canon = TablePlaneCanonicalizer(train_rest_z=0.025, snap_tol=0.01)
    delta = canon.delta(bottom_z, top_z)
    env.z_canonicalizer = canon
    corrected = env._build_obs()
    # 9-dim layout: ee_xyz(0:3), grip(3), top_xyz(4:7), bottom_xy(7:9)
    assert np.isclose(corrected[2], raw[2] + delta)   # ee_z shifted
    assert np.isclose(corrected[6], raw[6] + delta)   # top_z shifted
    assert np.isclose(corrected[0], raw[0])           # ee_x untouched
    assert np.isclose(corrected[3], raw[3])           # gripper untouched
    assert np.allclose(corrected[7:9], raw[7:9])      # bottom_xy untouched


def test_scripted_randomizes_hover_and_place_within_range():
    cfg_env = gym.make(FAKE_ENV_ID)
    sp = ScriptedStackPolicy(cfg_env.unwrapped.config, mediocre=True)
    hovers, places = [], []
    for s in range(20):
        sp.reset(np.random.default_rng(s))
        hovers.append(sp._hover_height)
        places.append(sp._place_offset)
    assert all(0.10 <= h <= 0.25 for h in hovers)
    assert all(0.002 <= p <= 0.015 for p in places)
    assert len(set(hovers)) > 1 and len(set(places)) > 1   # actually varying


def test_scripted_randomization_can_be_disabled():
    from mile_franka.policies.scripted import ScriptedPolicyConfig
    sp = ScriptedStackPolicy(cfg=ScriptedPolicyConfig(randomize_heights=False), mediocre=False)
    sp.reset(np.random.default_rng(0))
    assert sp._hover_height == ScriptedPolicyConfig().hover_height
    assert sp._place_offset == 0.0
