#!/usr/bin/env python3
"""Tune MILE intervention cost / cdf_scale for a given policy and dataset.

Two phases:

  Phase 1 — Policy analysis.  Sample the policy's action distribution, measure
  the log-probability scale, and recommend (cost, cdf_scale) values that give a
  useful dynamic range for the intervention probability.  This phase only needs
  the base policy (no dataset, no mental model).

  Phase 2 — Grid search (optional).  Sweep (cost, cdf_scale) pairs and measure
  BCE + a discrimination score against recorded intervention labels.  When the
  mental model equals the policy the sweep is *degenerate* (P(ν=1) is nearly
  constant for every state) — the script detects this and prints a warning.

Typical usage::

  # Before any training — just analyze the policy and get recommendations:
  python scripts/tune_intervention_cost.py --policy trained_models/franka/base_policy --analyze_only

  # After one MILE round — refine with a trained mental model:
  python scripts/tune_intervention_cost.py                                  \
      --policy      trained_models/franka/base_policy                       \
      --mental_model output_dir/franka/mental_model                         \
      --dataset     output_dir/franka/accumulated_dataset_round0.pkl
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributions as D
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.sac.policies import SACPolicy


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _trusted_torch_load_compat():
    orig_load = torch.load
    torch.load = functools.partial(orig_load, weights_only=False)
    return orig_load


def _restore_torch_load(orig_load):
    torch.load = orig_load


def _parse_grid(spec: str) -> list[float]:
    """Parse comma-separated values or start:stop:step inclusive grid."""
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        if step <= 0:
            raise ValueError("grid step must be positive")
        n = int(math.floor((stop - start) / step)) + 1
        return [start + i * step for i in range(max(0, n))]
    return [float(x) for x in spec.split(",") if x.strip()]


def _load_policy(path: str, policy_type: str):
    cls = {
        "bc": ActorCriticPolicy,
        "sac": SACPolicy,
        "qnetwork": QNetwork,
    }[policy_type]
    return cls.load(path)


def _batched(xs: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


# ---------------------------------------------------------------------------
# Phase 1 — policy log-probability analysis
# ---------------------------------------------------------------------------

def analyze_policy_log_prob_scale(
    policy,
    num_states: int = 200,
    num_mc_samples: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Return log-probability statistics and recommended (cost, cdf_scale).

    Creates dummy states (zeros) — the policy's feature extractor + RunningNorm
    will normalize them, so the absolute values don't matter for measuring the
    *scale* of log-probabilities.  The key question is: what is the typical
    magnitude and spread of ln π(a|s) for this policy?

    Returns a dict with keys:
      logp_mean, logp_std, logp_p05, logp_p95  — log-prob distribution stats
      recommended_cost, recommended_scale       — suggested COST_LOOKUP values
      baseline_p_int                            — P(ν=1) when policy ≈ mental model
    """
    policy.eval()
    # Use fake zero-states; RunningNorm will squash them to ~N(0,1) which is fine
    # for measuring the output-distribution scale.
    obs_dim = policy.observation_space.shape[0]
    dummy = torch.zeros((num_states, obs_dim), device=device)

    with torch.no_grad():
        dist = policy.get_distribution(dummy)
        pd = dist.distribution  # DiagGaussianDistribution
        mu = pd.loc             # (num_states, action_dim)
        std = pd.scale          # (num_states, action_dim)

        # MC samples from the policy
        normal = D.Normal(mu.unsqueeze(0), std.unsqueeze(0))
        samples = normal.sample((num_mc_samples,))        # (M, N, D)
        log_probs = normal.log_prob(samples).sum(dim=-1)  # (M, N)  summed over action dims

        # Statistics over all (sample, state) pairs
        flat = log_probs.flatten().cpu().numpy()
        stats = {
            "action_dim": mu.shape[-1],
            "logp_mean": float(np.mean(flat)),
            "logp_std": float(np.std(flat)),
            "logp_p05": float(np.percentile(flat, 5)),
            "logp_p95": float(np.percentile(flat, 95)),
            "logp_min": float(np.min(flat)),
            "logp_max": float(np.max(flat)),
        }

        # ---- recommend (cost, scale) ----
        # When policy ≈ mental_model:
        #   ln π(a) − E[ln π(a′)] ≈ 0  (both draw from same distribution)
        #   P(ν=1) = Φ((0 − cost) / scale) = Φ(−cost / scale)
        #
        # We want baseline P(int) around 0.15–0.25 so there is room to go both
        # lower (ν=0 steps) and higher (ν=1 steps).  Target Φ(−c/σ) ≈ 0.2,
        # which means c/σ ≈ 0.84.
        #
        # The *scale* σ should be small enough that a meaningful divergence
        # between policy and mental model (Δ ≈ 2–4 in log-prob space) moves
        # P(int) substantially.  With σ = 2 and Δ = 2:
        #   P(int) = Φ((2 − cost) / 2)  →  if cost=2, P(int) goes from
        #   Φ(−1)=0.16 to Φ(0)=0.50.
        #
        # We set σ ≈ logp_std (the natural variation across MC samples) so that
        # the CDF argument changes by ~1σ when the mental model diverges by one
        # natural unit.  Then cost = 0.84 × σ for baseline ≈ 0.2.
        logp_std = stats["logp_std"]
        recommended_scale = max(0.5, round(logp_std * 2.0) / 2.0)  # round to 0.5
        recommended_cost = max(0.5, round(0.84 * recommended_scale * 2.0) / 2.0)

        baseline_p_int = float(D.Normal(0.0, 1.0).cdf(
            torch.tensor(-recommended_cost / recommended_scale)
        ))

        stats.update({
            "recommended_cost": recommended_cost,
            "recommended_scale": recommended_scale,
            "baseline_p_int": baseline_p_int,
        })
        return stats


# ---------------------------------------------------------------------------
# Phase 2 — grid search
# ---------------------------------------------------------------------------

def _predict_probs(
    states: np.ndarray,
    policy,
    mental_model,
    cost: float,
    scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probs: list[np.ndarray] = []
    policy.eval()
    mental_model.eval()
    with torch.no_grad():
        for batch in _batched(states, batch_size):
            state_t = torch.as_tensor(batch, dtype=torch.float32, device=device)
            from mile.computational_model import computational_intervention_model
            _, _, intervention_prob, _, _ = computational_intervention_model(
                state=state_t,
                mental_model=mental_model,
                policy=policy,
                cost=cost,
                cdf_scale=scale,
            )
            probs.append(intervention_prob[:, 1].detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    eps = 1e-7
    p = np.clip(p, eps, 1.0 - eps)
    pred_rate = float(p.mean())
    actual_rate = float(y.mean())

    # Discrimination: difference in mean P(int) between ν=1 and ν=0 steps
    mask_1 = y == 1
    mask_0 = y == 0
    p1_mean = float(p[mask_1].mean()) if mask_1.any() else 0.0
    p0_mean = float(p[mask_0].mean()) if mask_0.any() else 0.0
    discrimination = p1_mean - p0_mean

    return {
        "bce": float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "actual_rate": actual_rate,
        "pred_rate": pred_rate,
        "rate_abs_err": abs(pred_rate - actual_rate),
        "discrimination": discrimination,
        "p_intervene_min": float(p.min()),
        "p_intervene_p50": float(np.percentile(p, 50)),
        "p_intervene_p90": float(np.percentile(p, 90)),
        "p_intervene_max": float(p.max()),
        "p_intervene_var": float(p.max() - p.min()),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tune MILE intervention cost / cdf_scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dataset", default="output_dir/franka/accumulated_dataset_round0.pkl")
    ap.add_argument("--policy", default="trained_models/franka/base_policy")
    ap.add_argument("--mental_model", default=None,
                    help="Mental model path (default: same as --policy)")
    ap.add_argument("--policy_type", choices=["bc", "sac", "qnetwork"], default="bc")
    ap.add_argument("--cost_grid", default="0.5,1,1.5,2,3,5,10,20,50,100,150",
                    help="comma list or start:stop:step, e.g. 0.5,1,2,5 or 0:10:2")
    ap.add_argument("--scale_grid", default="0.5,1,1.5,2,3,5,10,25,50,100,150",
                    help="comma list or start:stop:step")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--objective", choices=["bce", "brier", "rate_abs_err",
                                             "discrimination", "composite"],
                    default="composite")
    ap.add_argument("--disc_weight", type=float, default=2.0,
                    help="Weight of discrimination term in composite score "
                         "(higher = prefer parameters that separate ν=0 from ν=1)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--analyze_only", action="store_true",
                    help="Only run Phase 1 (policy analysis), skip grid search")
    ap.add_argument("--no_grid_search", action="store_true",
                    help="Skip Phase 2 grid search")
    args = ap.parse_args()

    device = torch.device(args.device)
    orig_load = _trusted_torch_load_compat()
    try:
        policy = _load_policy(args.policy, args.policy_type).to(device)
        mental_path = args.mental_model or args.policy
        mental_model = _load_policy(mental_path, args.policy_type).to(device)
    finally:
        _restore_torch_load(orig_load)

    same_weights = (args.mental_model is None) or (args.mental_model == args.policy)

    # ── Phase 1: policy analysis ────────────────────────────────────────
    print("=" * 72)
    print("Phase 1 — Policy log-probability analysis")
    print("=" * 72)
    stats = analyze_policy_log_prob_scale(policy, device=device)
    print(f"  Action dim:            {stats['action_dim']}")
    print(f"  log p(a|s)  mean:      {stats['logp_mean']:+.4f}")
    print(f"  log p(a|s)  std:       {stats['logp_std']:.4f}")
    print(f"  log p(a|s)  5th–95th:  [{stats['logp_p05']:+.4f}, {stats['logp_p95']:+.4f}]")
    print(f"  log p(a|s)  range:     [{stats['logp_min']:+.4f}, {stats['logp_max']:+.4f}]")
    print()
    print(f"  Recommended cost:      {stats['recommended_cost']:.1f}")
    print(f"  Recommended cdf_scale: {stats['recommended_scale']:.1f}")
    print(f"  → baseline P(ν=1):     {stats['baseline_p_int']:.3f}  (when policy ≈ mental model)")
    print()
    print(f"  Config override:")
    print(f'    "intervention_cost":      {stats["recommended_cost"]:.1f},')
    print(f'    "intervention_cdf_scale": {stats["recommended_scale"]:.1f},')
    print()

    if args.analyze_only:
        return

    # ── Phase 2: grid search ────────────────────────────────────────────
    if args.no_grid_search:
        return

    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)
    states = np.asarray(dataset["state"], dtype=np.float32)
    y = np.asarray(dataset["intervention"], dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"expected state array (N,D), got {states.shape}")
    if len(states) != len(y):
        raise ValueError(f"state/intervention length mismatch: {len(states)} vs {len(y)}")

    costs = _parse_grid(args.cost_grid)
    scales = _parse_grid(args.scale_grid)

    print("=" * 72)
    print("Phase 2 — Grid search")
    print("=" * 72)
    if same_weights:
        print("⚠️  WARNING: mental_model == policy — P(ν=1) is nearly CONSTANT for all")
        print("    (cost, scale) pairs.  The grid search CANNOT find meaningful parameters")
        print("    in this regime.  Use the Phase 1 recommendations above, or provide a")
        print("    trained mental model via --mental_model.")
        print()
    print(f"dataset={args.dataset}  n={len(y)}  intervention_rate={float(y.mean()):.4f}")
    print(f"policy={args.policy}  mental_model={mental_path}  device={device}")
    print(f"grid: {len(costs)} costs × {len(scales)} scales = {len(costs) * len(scales)}")
    print()

    rows: list[dict[str, float]] = []
    max_p_var = 0.0
    for scale in scales:
        for cost in costs:
            p = _predict_probs(states, policy, mental_model, cost, scale,
                               args.batch_size, device)
            row = {"cost": float(cost), "cdf_scale": float(scale), **_metrics(y, p)}
            rows.append(row)
            if row["p_intervene_var"] > max_p_var:
                max_p_var = row["p_intervene_var"]

    # Composite score: lower is better.
    #   composite = bce − disc_weight × discrimination
    # We want low BCE AND high discrimination.
    for r in rows:
        r["composite"] = r["bce"] - args.disc_weight * max(0.0, r["discrimination"])

    objective = args.objective
    rows.sort(key=lambda r: (r[objective], r["rate_abs_err"], r["bce"]))

    # ── Degeneracy check ────────────────────────────────────────────────
    if max_p_var < 0.01:
        print("⚠️  DEGENERATE: max Pvar across all grid points =",
              f"{max_p_var:.6f} (< 0.01)")
        print("    The intervention model produces effectively constant output for every")
        print("    state because policy ≈ mental_model.  The 'best' parameters below just")
        print("    match the constant to the average intervention rate — they will NOT work")
        print("    for training (no gradient signal).")
        print()
        print("    → Use the Phase 1 recommended values instead:")
        print(f'      "intervention_cost":      {stats["recommended_cost"]:.1f},')
        print(f'      "intervention_cdf_scale": {stats["recommended_scale"]:.1f},')
        print()

    best = rows[0]
    print(f"Best by {objective}:")
    print(json.dumps(best, indent=2, sort_keys=True))
    print()
    print(f"Top {args.top_k} candidates:")
    header = (f"{'cost':>7s}  {'scale':>7s}  {'bce':>8s}  {'disc':>7s}  "
              f"{'composite':>9s}  {'Pvar':>8s}  {'rate':>10s}")
    print(header)
    print("-" * len(header))
    for row in rows[: args.top_k]:
        print(
            f"{row['cost']:7.2f}  {row['cdf_scale']:7.2f}  "
            f"{row['bce']:8.5f}  {row['discrimination']:+7.4f}  "
            f"{row['composite']:+9.5f}  {row['p_intervene_var']:8.6f}  "
            f"{row['pred_rate']:5.4f}/{row['actual_rate']:5.4f}"
        )

    print()
    print("Config override for best:")
    print(json.dumps({
        "intervention_cost": best["cost"],
        "intervention_cdf_scale": best["cdf_scale"],
    }, indent=2))


if __name__ == "__main__":
    main()
