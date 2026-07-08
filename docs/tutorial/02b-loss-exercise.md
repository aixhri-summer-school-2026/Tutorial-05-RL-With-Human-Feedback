# The MILE Loss Function

## The Problem

MILE jointly trains two networks from intervention data. The training signal comes from a single combined loss: one term for predicting *when* the human intervenes, and one term for predicting *what* the human does when they do.

**File to edit:**
```
mile_franka/tutorial/loss_exercise.py
```

## Your Task

Implement:
```python
def mile_cont_loss_fn(
    intervention_prob,       # (batch, 2)  — softmax [p(ν=0), p(ν=1)]
    mu,                      # (batch, action_dim) — predicted action mean
    log_std,                 # (batch, action_dim) — predicted action log-std
    ground_truth_action,     # (batch, action_dim) — human's action
    ground_truth_intervention,  # (batch,) long — binary flag (0 or 1)
) -> tuple[Tensor, Tensor, Tensor]:  # (loss, continuous_loss, discrete_loss)
```

## The Loss

### Discrete term — intervention prediction

The intervention head predicts ν ∈ {0, 1} as a 2-class problem:

```
discrete_loss = NLL(log intervention_prob, ground_truth_intervention)
```

`NLL` here is `F.nll_loss` applied to log-probabilities. Clamp `intervention_prob` to `[1e-7, 1-1e-7]` before taking the log to avoid `log(0)`.

> **Note on BCE vs. NLL:** The concepts doc calls this "BCE." The 2-class NLL on a softmax head is mathematically equivalent to BCE on the positive class — we use the 2-class form because MILE's intervention head outputs `[p(ν=0), p(ν=1)]`.

### Continuous term — action imitation

When the human intervenes (ν=1), the policy is trained to match the human's action under a Gaussian NLL:

```
continuous_loss = -mean( log N(ground_truth_action | μ, exp(log_std)) )
                                [for ν==1 steps only]
```

If no rows satisfy `ground_truth_intervention == 1` and `intervention_prob[:, -1] > 0.0`, return `continuous_loss = tensor(0.0)`.

### Combined loss

```
loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss
```

`LAMBDA1` and `LAMBDA2` are defined at module level in the exercise file.

## Design Questions

1. **Why mask on ν=1?** When the human doesn't intervene, what do we know about the human's action — and why does that make it unusable as a training signal?

2. **Why clamp before log?** What goes wrong numerically if `intervention_prob` contains an exact 0 or 1?

3. **Action dimensions:** Each action dimension is modeled as an independent Gaussian. How does that change how you sum `log_prob` across the action dim?

4. **The two terms pull in different directions.** The discrete term trains the intervention head (both networks). The continuous term trains only the policy (ν=1 steps only). What would happen if you weighted one term much higher than the other?

## Testing Your Implementation

```bash
make tutorial-check-loss
```

The tests check:
- Reference solution produces known output values
- Your implementation matches the reference on the same batch
- Continuous loss is exactly 0.0 when all `ground_truth_intervention == 0`

## The Answer Key

If your implementation is incomplete when you run `make tutorial-metaworld` or `make tutorial-collect-train`, the training harness will detect this, print:
```
[tutorial] your loss is missing/incorrect — applying the answer key so you can train.
```
and automatically apply `mile_franka/tutorial/_loss_solution.py`. You can train either way, but we encourage you to implement it yourself first.

## Why the ν=1 Mask Matters

In MILE, the human's action is only meaningful when they actually intervene. If ν=0, the human let the robot run — we see no action from them, so there is nothing to imitate. Masking ensures:

1. We only compute the imitation loss on real intervention examples.
2. Batches with zero interventions contribute `continuous_loss = 0.0`, which is correct.

The discrete term, by contrast, uses *all* timesteps — every step (intervened or not) is a training signal for the intervention prediction head.

## Next Step

→ **[02c-intervention-model.md](02c-intervention-model.md)**: Exercise 3 — design p(ν=1 | state).
