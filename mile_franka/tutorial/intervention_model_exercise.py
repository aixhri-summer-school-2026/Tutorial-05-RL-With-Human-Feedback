"""MILE intervention model exercise — IMPLEMENT THIS.

The MILE training loop needs to know: given the current state, how likely is it
that a human would intervene?  This is p(ν=1 | state) — the intervention probability.

Your job: implement compute_intervention_prob.

Inputs:
- state:        (batch, state_dim) float tensor — the current observation.
- mental_model: ActorCriticPolicy — the human's model of what the robot will do.
- policy:       ActorCriticPolicy — what the robot is actually planning to do.

Returns:
- intervention_prob: (batch,) float tensor, values in [0, 1].
  Each entry is the probability that a human would intervene at that state.

You have access to both the policy and the mental model. Both are
ActorCriticPolicy objects — you can call policy.get_distribution(state)
to get a Gaussian distribution over actions.

Some questions to guide your design:
  - When would a human want to take over?
  - What does it mean for the robot to "need help"?
  - How can you use the two distributions (policy vs mental_model) to answer this?

Hint: you might compute the entropy of the policy, the KL divergence between
the two distributions, or something else entirely. There is no single right answer
until you see ours at the end.

Run `make tutorial-check-intervention-model` to test your implementation.
If you run out of time, the answer key applies automatically when you train.
"""
import torch
import torch.distributions as D

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_intervention_prob(state, mental_model, policy):
    # TODO: implement your intervention model.
    # Return a (batch,) float tensor with values in [0, 1].

    raise NotImplementedError(
        "Implement compute_intervention_prob — see 02b-intervention-model.md for guidance."
    )
