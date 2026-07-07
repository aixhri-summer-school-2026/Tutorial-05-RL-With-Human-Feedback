# Design Your Own Intervention Model

## The Problem

MILE's training loop needs one key input: **p(ν=1 | state)** — the probability that a human would intervene at the current state.

If you have this function, everything else follows:
- The loss function you implemented uses it to weight the policy update.
- The training loop uses it to decide when to take over during rollouts.

**But how do you compute p(ν=1 | state)?**

You have two policy networks:
- **`policy`**: the robot's current action plan (what the robot will actually do).
- **`mental_model`**: the human's mental model of the robot (what the human *expects* the robot to do).

Both are `ActorCriticPolicy` objects. You can call `policy.get_distribution(state)` on either to get a Gaussian distribution over actions.

**File to edit:**
```
mile_franka/tutorial/intervention_model_exercise.py
```

## Your Task

Implement:
```python
def compute_intervention_prob(state, mental_model, policy):
    # state: (batch, state_dim)
    # mental_model, policy: ActorCriticPolicy
    # returns: (batch,) float tensor, values in [0, 1]
```

## Design Questions

Before jumping to code, think through:

1. **When does a human want to intervene?** What signal would tell you "the robot is about to fail"?

2. **What can you compute from two distributions?**
   - Entropy of the policy distribution (high entropy = uncertain robot)?
   - KL divergence between `mental_model` and `policy`?
   - Something else?

3. **What does "the policy disagrees with the mental model" look like mathematically?**

## Useful Building Blocks

```python
# Get the policy's action distribution at a given state
dist_obj = policy.get_distribution(state)
policy_dist = dist_obj.distribution          # D.Normal (Gaussian)
mu = policy_dist.loc                         # mean: (batch, action_dim)
std = policy_dist.scale                      # std:  (batch, action_dim)

# Sample actions from the distribution
samples = policy_dist.rsample((1000,))       # (1000, batch, action_dim)

# Log-probability of an action under the distribution
log_p = policy_dist.log_prob(samples)        # (1000, batch, action_dim)

# Sum across action dimensions (actions are independent per dimension)
from mile.computational_model import sum_independent_dims
scalar_log_p = sum_independent_dims(log_p)  # (1000, batch)

# monte_carlo_samples utility (already available)
from mile.computational_model import monte_carlo_samples
samples = monte_carlo_samples(state=state, policy=policy, num_samples=1000)
```

## Validation

After implementing, check your output shape and range:
```python
import torch
# state: (batch, state_dim), both policies loaded
out = compute_intervention_prob(state, mental_model, policy)
assert out.shape == (batch_size,), f"Expected (batch,), got {out.shape}"
assert (out >= 0).all() and (out <= 1).all(), "Values must be in [0, 1]"
```

Run the check:
```bash
make tutorial-check-intervention-model
```

## Answer Key

If your implementation is not yet complete when you run `make tutorial-metaworld` or
`make tutorial-collect-train`, the training harness will automatically apply the
reference solution and print:
```
[tutorial] intervention model not yet implemented — applying the answer key so you can train.
```

You can still train and collect interventions even with the answer key active.

## The Reveal

After you've tried designing your own model, we'll show you how MILE actually computes
p(ν=1 | s). The key idea:

> A human intervenes when the robot's planned actions *diverge* from what the human
> expected the robot to do.

MILE measures this divergence using the **robot's policy log-probability**:
- Sample 1000 actions from the *mental model* (what human expects).
- Compute the robot policy's log-prob of those samples → baseline log-likelihood.
- Sample 1000 actions from the *robot policy*.
- For each: is this action's log-prob above or below the baseline?
- Pass the signed gap through a Normal CDF → per-sample intervention probability.
- Average across samples → p(ν=1 | s).

The full implementation is in `mile_franka/tutorial/_intervention_model_solution.py` —
read it after you've tried your own approach.

## Why This Matters

The computational model of human intervention is what makes MILE different from
standard imitation learning. Without a model of *when* humans intervene, you're
just doing BC on a noisy dataset. With a good intervention model, the algorithm
learns from *both* the interventions (what the human corrected) and the
non-interventions (what the robot was allowed to do on its own) — using the
**absence of intervention** as a signal that the policy was acceptable.

## Next Step

Once the check passes (or you're ready to move on):
→ **[03-walkthrough.md](03-walkthrough.md)** — run MetaWorld with all three exercises active.
