"""Scripted intervener exercise — IMPLEMENT THIS.

When MILE collects training data with a synthetic "human", something has to decide:
given what the expert would do vs. what the base policy is doing, should the human
step in right now?

This is the scripted version of the same question as compute_intervention_prob — but
here you return a hard binary decision (bool) instead of a probability.

Inputs:
- expert_action: np.ndarray, shape (4,) — what the scripted expert wants to do.
                 Layout: [dx, dy, dz, gripper_command]
- rollout_action: np.ndarray, shape (4,) — what the base policy is currently doing.
                  Same layout as expert_action.

Return:
- bool — True to intervene (take over), False to let the policy continue.

Design questions:
  - What does it mean for the expert and the policy to "disagree"?
  - Are all action dimensions equally important?
  - Does the gripper command warrant special treatment?
  - What threshold makes sense for translation drift?

Run `make tutorial-check-scripted-intervener` to test your implementation.
If you don't implement this, the reference rule (L2 norm + gripper check) is used.
"""
import numpy as np


def should_intervene(expert_action: np.ndarray, rollout_action: np.ndarray) -> bool:
    # TODO: return True if the human should step in, False otherwise.
    raise NotImplementedError(
        "Implement should_intervene — see docs/tutorial/02c-scripted-intervener.md"
    )
