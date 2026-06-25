"""MILE loss exercise — IMPLEMENT THIS.

Implement the continuous MILE loss used to train the policy and the mental model.

Inputs (all torch tensors):
- intervention_prob: (batch, 2) — p(nu=0), p(nu=1) from the intervention model.
- mu:                (batch, action_dim) — predicted action mean.
- log_std:           (batch, action_dim) — predicted action log-std.
- ground_truth_action: (batch, action_dim) — the HUMAN action.
- ground_truth_intervention: (batch,) — nu in {0,1} (long).

Return a 3-tuple: (loss, continuous_loss, discrete_loss) where
- discrete_loss = NLL of the intervention head vs ground_truth_intervention
  (clamp probs to [1e-7, 1-1e-7]; use F.nll_loss on log-probs).
- continuous_loss = Gaussian NLL of ground_truth_action under Normal(mu, exp(log_std)),
  averaged, computed ONLY on rows where nu==1 (and intervention_prob[:,-1] > 0).
  If there are no such rows, continuous_loss = tensor(0.0).
- loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss.

Run `make tutorial-check-loss` until it is green. If you run out of time, the answer
key is applied automatically when you train.
"""
import torch
import torch.distributions as D
import torch.nn.functional as F

from stable_baselines3.common.distributions import sum_independent_dims

device = "cuda" if torch.cuda.is_available() else "cpu"


def mile_cont_loss_fn(intervention_prob, mu, log_std, ground_truth_action,
                      ground_truth_intervention, LAMBDA1=1.0, LAMBDA2=1.0,
                      reduction="mean"):
    # Discrete loss: NLL of the intervention head
    pred_probs = torch.clamp(intervention_prob, 1e-7, 1 - 1e-7)
    discrete_loss = F.nll_loss(torch.log(pred_probs), ground_truth_intervention,
                               reduction=reduction)

    # Continuous loss: Gaussian NLL on ν=1 steps only
    idx = torch.logical_and(ground_truth_intervention == 1,
                            intervention_prob[:, -1] > 0.0)
    if idx.sum() == 0:
        continuous_loss = torch.tensor(0.0).to(device)
    else:
        dist = D.Normal(mu[idx], log_std[idx].exp())
        log_prob = sum_independent_dims(dist.log_prob(ground_truth_action[idx]))
        continuous_loss = -log_prob.mean()

    loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss
    return loss, continuous_loss, discrete_loss
