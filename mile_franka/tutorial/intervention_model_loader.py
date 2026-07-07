"""Resolve which intervention model to use: the participant's exercise if it runs
and returns a valid (batch,) tensor, else the reference answer key (auto-apply).
Used by scripts/tutorial_train.py to monkeypatch
mile.computational_model.computational_intervention_model.
"""
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy

from mile.computational_model import LOG_STD_MIN, LOG_STD_MAX

try:
    from mile_franka.tutorial import _intervention_model_solution as _solution
    _HAS_SOLUTION = True
except ImportError:
    _solution = None
    _HAS_SOLUTION = False


def _make_full_model(prob_fn):
    """Wrap a compute_intervention_prob(state, mental_model, policy) -> (batch,) function
    into the full computational_intervention_model(state, mental_model, policy, cost, cdf_scale)
    signature expected by mile.algorithm.InterventionTrainer."""

    def wrapped(state, mental_model, policy, cost=0, cdf_scale=1.0):
        intervention_prob_raw = prob_fn(state, mental_model, policy)  # (batch,)
        intervention_prob_raw = torch.clamp(intervention_prob_raw.float(), 1e-6, 1.0 - 1e-6)

        # Get policy distribution parameters
        if isinstance(policy, SACPolicy):
            mu, log_std, _ = policy.actor.get_action_dist_params(state)
        elif isinstance(policy, ActorCriticPolicy):
            dist_obj = policy.get_distribution(state)
            pd = dist_obj.distribution
            mu = pd.loc
            log_std = torch.log(pd.scale)
        else:
            raise ValueError("policy must be SACPolicy or ActorCriticPolicy")

        # Compute intervention-weighted policy params (same as original MILE model)
        final_mu = intervention_prob_raw.unsqueeze(1) * mu
        final_log_std = torch.log(intervention_prob_raw.unsqueeze(1)) + log_std
        final_log_std = torch.clamp(final_log_std, LOG_STD_MIN, LOG_STD_MAX)

        # Convert (batch,) to (batch, 2): [p(ν=0), p(ν=1)]
        ip = intervention_prob_raw.unsqueeze(-1)
        intervention_prob = torch.cat((1 - ip, ip), dim=-1)

        return final_mu, final_log_std, intervention_prob, mu, log_std

    return wrapped


def _make_sample_state(device="cpu"):
    """Build a tiny (4, 8) state tensor for validation."""
    g = torch.Generator().manual_seed(42)
    return torch.randn(4, 8, generator=g, device=device)


def _is_implemented(fn):
    """Return True if fn does not immediately raise NotImplementedError on a trivial call."""
    try:
        import inspect
        src = inspect.getsource(fn)
        # Unimplemented scaffold: contains `raise NotImplementedError`
        if "raise NotImplementedError" in src:
            return False
        return True
    except Exception:
        return False


def resolve_intervention_model():
    """Return (full_model_fn, used_answer_key: bool).

    full_model_fn has the computational_intervention_model(state, mental_model, policy, cost, cdf_scale)
    signature and can be used to monkeypatch mile.algorithm.
    """
    from mile_franka.tutorial import intervention_model_exercise as exercise

    if _is_implemented(exercise.compute_intervention_prob):
        print("[tutorial] using YOUR intervention model ✔")
        return _make_full_model(exercise.compute_intervention_prob), False

    if _HAS_SOLUTION:
        print("[tutorial] intervention model not yet implemented — applying the answer key so you can train.")
        return _make_full_model(_solution.compute_intervention_prob), True

    print("[tutorial] intervention model not yet implemented (no answer key available — implement the exercise).")
    return _make_full_model(exercise.compute_intervention_prob), False
