"""Reference answer for the loss exercise — identical to mile.algorithm.mile_cont_loss_fn."""
import torch
import torch.distributions as D
import torch.nn.functional as F

from stable_baselines3.common.distributions import sum_independent_dims

device = "cuda" if torch.cuda.is_available() else "cpu"


def _disc_loss(pred_probs, gt_labels, reduction="mean"):
    pred_probs = torch.clamp(pred_probs, 1e-7, 1 - 1e-7)
    return F.nll_loss(torch.log(pred_probs), gt_labels, reduction=reduction)


def mile_cont_loss_fn(intervention_prob, mu, log_std, ground_truth_action,
                      ground_truth_intervention, LAMBDA1=1.0, LAMBDA2=1.0,
                      reduction="mean"):
    discrete_loss = _disc_loss(intervention_prob, ground_truth_intervention, reduction=reduction)
    idx = torch.logical_and(ground_truth_intervention == 1, intervention_prob[:, -1] > 0.0)
    if idx.sum() == 0:
        continuous_loss = torch.tensor(0.0).to(device)
    else:
        dist = D.Normal(mu[idx], log_std[idx].exp())
        log_prob = sum_independent_dims(dist.log_prob(ground_truth_action[idx]))
        continuous_loss = -log_prob.mean()
    loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss
    return loss, continuous_loss, discrete_loss
