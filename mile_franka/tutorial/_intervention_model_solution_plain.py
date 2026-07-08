"""Reference answer for the intervention model exercise."""
import torch
import torch.distributions as D

device = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1000


def compute_intervention_prob(state, mental_model, policy):
    """MILE probit model: p(ν=1|s) = E_{a~π}[Φ(log π(a|s) − E_{â~π̂}[log π(â|s)])]"""
    with torch.no_grad():
        dist_policy = policy.get_distribution(state).distribution
        dist_mental = mental_model.get_distribution(state).distribution

        # Baseline: mean log-prob of mental-model samples under the robot policy.
        mental_samples = dist_mental.sample((N_SAMPLES,))               # (N, B, A)
        log_pi_mental = dist_policy.log_prob(mental_samples).sum(-1)    # (N, B)
        baseline = log_pi_mental.mean(0)                                # (B,)

        # For each policy sample compute the gap vs the baseline.
        policy_samples = dist_policy.sample((N_SAMPLES,))               # (N, B, A)
        log_pi_policy = dist_policy.log_prob(policy_samples).sum(-1)    # (N, B)
        gap = log_pi_policy - baseline.unsqueeze(0)                     # (N, B)

        # Standard-normal CDF of the gap → per-sample intervention probability.
        cdf = D.Normal(0.0, 1.0).cdf(gap)                              # (N, B)
        return cdf.mean(0)                                              # (B,)