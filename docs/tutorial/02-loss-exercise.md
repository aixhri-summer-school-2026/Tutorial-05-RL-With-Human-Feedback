# Task G3: The MILE Loss Function

## The Problem

You will implement `mile_cont_loss_fn` — the combined loss used to train both the **policy** and the **mental model** in MILE.

**File to edit:**
```
mile_franka/tutorial/loss_exercise.py
```

In this file, the function `mile_cont_loss_fn` is a stub. Replace the `raise NotImplementedError(...)` line with your implementation.

## The Loss

The MILE loss is a weighted combination of two terms:

### 1. Discrete Loss: Intervention Prediction

The intervention head predicts whether a human will intervene (ν = 1) or not (ν = 0) at each timestep. This is a 2-class classification problem.

**Formula:**
```
discrete_loss = NLL(log P(ν | state), ground_truth_ν)
```

Where:
- `intervention_prob`: shape `(batch, 2)` — the softmax output `[p(ν=0), p(ν=1)]`
- `ground_truth_intervention`: shape `(batch,)` — the binary label (0 or 1)
- NLL = Negative Log Likelihood (F.nll_loss on log-probabilities)

**Implementation notes:**
- Clamp `intervention_prob` to `[1e-7, 1 - 1e-7]` to avoid log(0)
- Take the log: `log(intervention_prob)`
- Pass to `F.nll_loss()` with the ground truth labels

> **Note on BCE vs. NLL:** The concepts doc calls this loss "binary cross-entropy (BCE)." Here we implement it as 2-class NLL using a softmax head (rather than a sigmoid). These are mathematically equivalent for binary classification — `NLL(softmax([p0, p1]), label)` equals `BCE(p1, label)`. We use the 2-class form because MILE's intervention head outputs a 2-class distribution, which `F.nll_loss` handles naturally.

### 2. Continuous Loss: Action Prediction

When the human does intervene (ν = 1), the mental model predicts the human's action. This is modeled as a Gaussian distribution.

**Formula:**
```
continuous_loss = -mean( log p(a_human | μ, σ) )  [for ν==1 steps only]
```

Where:
- `a_human` = `ground_truth_action`, shape `(batch, action_dim)`
- `μ` = `mu`, the predicted action mean, shape `(batch, action_dim)`
- `log(σ)` = `log_std`, the predicted action log-standard-deviation, shape `(batch, action_dim)`
- The normal distribution is parameterized as `N(μ, exp(log_std))`

**Key constraint:**
- **Only compute this loss on rows where `ground_truth_intervention == 1` AND `intervention_prob[:, -1] > 0.0`**
  - Why? We only care about predicting the human's action when they actually intervened.
  - If no rows satisfy this condition, return `continuous_loss = tensor(0.0)`

**Implementation notes:**
- Create a boolean mask: `idx = torch.logical_and(ground_truth_intervention == 1, intervention_prob[:, -1] > 0.0)`
- If `idx.sum() == 0`, set `continuous_loss = torch.tensor(0.0).to(device)` and skip the distribution
- Otherwise:
  - Use `torch.distributions.Normal(mu[idx], log_std[idx].exp())` to create a Gaussian
  - Compute `dist.log_prob(ground_truth_action[idx])` — this gives log-likelihood per dimension
  - Use `sum_independent_dims()` from `stable_baselines3.common.distributions` to sum across action dims
  - Take the negative mean: `continuous_loss = -log_prob.mean()`

### 3. Combined Loss

**Formula:**
```
loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss
```

Where `LAMBDA1` and `LAMBDA2` are scalar weights (default both 1.0).

## Tensor Shapes

| Variable | Shape | Dtype | Notes |
|----------|-------|-------|-------|
| `intervention_prob` | (batch, 2) | float | softmax probabilities for ν=0 and ν=1 |
| `mu` | (batch, action_dim) | float | predicted action mean |
| `log_std` | (batch, action_dim) | float | predicted action log-std |
| `ground_truth_action` | (batch, action_dim) | float | human's actual action |
| `ground_truth_intervention` | (batch,) | long | binary intervention flag (0 or 1) |
| **output: loss** | scalar | float | total loss |
| **output: continuous_loss** | scalar | float | -mean log p(action \| ν=1) |
| **output: discrete_loss** | scalar | float | NLL of intervention binary classification |

## Testing Your Implementation

After you finish, test with:
```bash
make tutorial-check-loss
```

**Expected output (before implementation):**
```
test_loss_exercise.py::test_candidate_matches_reference_when_implemented SKIPPED
  exercise not yet implemented (blank scaffold)
```

**Expected output (after correct implementation):**
```
test_loss_exercise.py::test_solution_matches_itself_known_values PASSED
test_loss_exercise.py::test_candidate_matches_reference_when_implemented PASSED
test_loss_exercise.py::test_continuous_term_only_uses_intervention_steps PASSED

=== 3 passed ===
```

If a test fails, read the error message carefully — it usually points to a specific mismatch (e.g., "continuous_loss computation is wrong" or "discrete_loss should use clamped probabilities").

## The Answer Key

If your implementation is incomplete or incorrect when you run `make tutorial-metaworld` or `make tutorial-collect-train`, the training script will detect this and **silently apply the reference solution** from `mile_franka/tutorial/_loss_solution.py`.

This means you never have to manually patch your code — the training harness handles it. However, **we encourage you to complete the implementation yourself** so you understand the loss before moving forward.

## Worked Explanation (After Attempting)

Here is the structure of the correct solution:

```python
def mile_cont_loss_fn(intervention_prob, mu, log_std, ground_truth_action,
                      ground_truth_intervention, LAMBDA1=1.0, LAMBDA2=1.0,
                      reduction="mean"):
    # Step 1: Discrete loss — NLL of the intervention binary classifier
    pred_probs = torch.clamp(intervention_prob, 1e-7, 1 - 1e-7)
    discrete_loss = F.nll_loss(torch.log(pred_probs), ground_truth_intervention, 
                               reduction=reduction)
    
    # Step 2: Continuous loss — Gaussian NLL, but ONLY for ν=1 steps
    # Create mask: intervention==1 AND intervention_prob[:,-1] > 0
    idx = torch.logical_and(ground_truth_intervention == 1, intervention_prob[:, -1] > 0.0)
    
    if idx.sum() == 0:
        # No intervention steps; no continuous loss to compute
        continuous_loss = torch.tensor(0.0).to(device)
    else:
        # Build a Gaussian distribution for the subset of intervention steps
        dist = D.Normal(mu[idx], log_std[idx].exp())
        # Compute log-likelihood of the human action under this distribution
        log_prob = sum_independent_dims(dist.log_prob(ground_truth_action[idx]))
        # Take the negative mean (NLL = -log-likelihood)
        continuous_loss = -log_prob.mean()
    
    # Step 3: Combine with weights
    loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss
    return loss, continuous_loss, discrete_loss
```

### Why the ν=1 Mask?

In MILE, the human's action is only meaningful when they intervene. If the human doesn't intervene (ν=0), we don't know what action they would have taken, so we shouldn't train on it. The mask ensures:

1. **We only see intervention examples** — `ground_truth_intervention == 1`
2. **We only predict when the model thinks it's plausible** — `intervention_prob[:, -1] > 0.0` (the log-space check prevents numerical issues)

If all samples are non-interventions (ν=0), the continuous loss is zero, which makes sense: no intervention data means no action predictions to make.

### Imports You'll Need

Your implementation should import (already present in the file):
- `torch`
- `torch.distributions as D`
- `torch.nn.functional as F`
- `sum_independent_dims` from `stable_baselines3.common.distributions`

The `device` variable is already defined at the module level.

## Next Steps

Once all three tests pass:
1. Commit your work (see main tutorial document for commit instructions)
2. Move on to **Task G4** (or the next task in your schedule)

If you get stuck or are short on time, no worries — the answer key will apply when you train, and you can learn by reviewing `mile_franka/tutorial/_loss_solution.py` afterward.
