"""Tests for the intervention model exercise.

These are lightweight shape/range checks — unlike the loss exercise, there is no
single "correct" implementation, so we only verify that the function:
1. Runs without crashing.
2. Returns a (batch,) float tensor.
3. Values are in [0, 1].

The reference solution is also tested to confirm the probit model itself is sane.
"""
import torch
import pytest
from unittest.mock import MagicMock


def _make_fake_policy(batch_size, action_dim=4, seed=0):
    """Return a mock ActorCriticPolicy-like object with get_distribution."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(batch_size, action_dim, generator=g)
    std = torch.ones(batch_size, action_dim)
    dist = torch.distributions.Normal(mu, std)
    mock_dist = MagicMock()
    mock_dist.distribution = dist
    policy = MagicMock()
    policy.get_distribution.return_value = mock_dist
    # Needed by monte_carlo_samples when called on an ActorCriticPolicy
    policy.action_space = MagicMock()
    policy.action_space.__class__ = type(None)  # make isinstance checks fail gracefully
    return policy


def _make_state(batch_size=4, state_dim=8, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, state_dim, generator=g)


# ── reference solution sanity ────────────────────────────────────────────────

def test_solution_returns_valid_tensor():
    """The reference probit model must return a (batch,) tensor in [0, 1]."""
    from mile_franka.tutorial._intervention_model_solution import compute_intervention_prob
    from mile_franka.tutorial.intervention_model_loader import _make_full_model
    from stable_baselines3.common.policies import ActorCriticPolicy
    from imitation.policies.base import NormalizeFeaturesExtractor
    from imitation.util.networks import RunningNorm
    import gymnasium as gym
    from stable_baselines3.common.utils import get_schedule_fn

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(8,))
    act_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=get_schedule_fn(1),
        net_arch=[32, 32],
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    )
    policy.set_training_mode(False)

    state = _make_state(batch_size=4, state_dim=8)
    with torch.no_grad():
        result = compute_intervention_prob(state, policy, policy)

    assert result.shape == (4,), f"Expected (4,), got {result.shape}"
    assert (result >= 0).all() and (result <= 1).all(), "Values must be in [0, 1]"
    assert torch.isfinite(result).all(), "Result must be finite"


# ── candidate exercise ───────────────────────────────────────────────────────

def _candidate():
    from mile_franka.tutorial.intervention_model_exercise import compute_intervention_prob
    return compute_intervention_prob


def test_candidate_skipped_when_not_implemented():
    try:
        fn = _candidate()
        # If it doesn't raise, just skip — we can't test shape without real policies
        pytest.skip("candidate raises NotImplementedError (not implemented yet)")
    except NotImplementedError:
        pytest.skip("exercise not yet implemented (blank scaffold)")


def test_candidate_shape_and_range_when_implemented():
    """If the student has implemented compute_intervention_prob, check shape and range."""
    import inspect
    fn = _candidate()
    src = inspect.getsource(fn)
    if "NotImplementedError" in src and "raise" in src:
        pytest.skip("exercise not yet implemented (blank scaffold)")

    from stable_baselines3.common.policies import ActorCriticPolicy
    from imitation.policies.base import NormalizeFeaturesExtractor
    from imitation.util.networks import RunningNorm
    import gymnasium as gym
    from stable_baselines3.common.utils import get_schedule_fn

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(8,))
    act_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=get_schedule_fn(1),
        net_arch=[32, 32],
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    )
    policy.set_training_mode(False)

    state = _make_state(batch_size=4, state_dim=8)
    with torch.no_grad():
        result = fn(state, policy, policy)

    assert result.shape == (4,), f"Expected (4,), got {result.shape}"
    assert (result >= 0).all() and (result <= 1).all(), "Values must be in [0, 1]"
    assert torch.isfinite(result).all(), "Result must be finite"
