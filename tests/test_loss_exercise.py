import numpy as np
import torch
import pytest

from mile_franka.tutorial._loss_solution import mile_cont_loss_fn as REF


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    n, a = 8, 4
    intervention_prob = torch.softmax(torch.randn(n, 2, generator=g), dim=-1)
    mu = torch.randn(n, a, generator=g)
    log_std = torch.zeros(n, a)
    gt_action = torch.randn(n, a, generator=g)
    gt_interv = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    return intervention_prob, mu, log_std, gt_action, gt_interv


def _candidate():
    from mile_franka.tutorial.loss_exercise import mile_cont_loss_fn as C
    return C


def test_solution_matches_itself_known_values():
    b = _batch()
    loss, cont, disc = REF(*b)
    assert torch.isfinite(loss)
    # disc loss = NLL of the 2-class intervention head; cont loss > 0 (some nu=1 steps)
    assert cont.item() > 0
    assert disc.item() > 0


def test_candidate_matches_reference_when_implemented():
    try:
        C = _candidate()
        out = C(*_batch())
    except NotImplementedError:
        pytest.fail("exercise not yet implemented — edit mile_franka/tutorial/loss_exercise.py")
    ref = REF(*_batch())
    for c, r in zip(out, ref):
        assert torch.allclose(c, r, atol=1e-6), "candidate loss != reference"


def test_continuous_term_only_uses_intervention_steps():
    # If all nu=0, continuous loss must be exactly 0.
    ip, mu, ls, ga, _ = _batch()
    gi = torch.zeros(8, dtype=torch.long)
    _, cont, _ = REF(ip, mu, ls, ga, gi)
    assert cont.item() == 0.0
