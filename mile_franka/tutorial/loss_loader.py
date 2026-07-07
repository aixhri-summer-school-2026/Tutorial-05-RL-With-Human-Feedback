"""Resolve which loss to train with: the participant's exercise if correct, else the
reference answer key (auto-apply). Used by scripts/tutorial_train.py to monkeypatch
mile.algorithm.mile_cont_loss_fn WITHOUT editing production code.
"""
import torch

try:
    from mile_franka.tutorial import _loss_solution as _solution
    _HAS_SOLUTION = True
except ImportError:
    _solution = None
    _HAS_SOLUTION = False


def _sample_batch():
    g = torch.Generator().manual_seed(0)
    ip = torch.softmax(torch.randn(8, 2, generator=g), dim=-1)
    mu = torch.randn(8, 4, generator=g)
    ls = torch.zeros(8, 4)
    ga = torch.randn(8, 4, generator=g)
    gi = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    return ip, mu, ls, ga, gi


def resolve_loss_fn():
    """Return (loss_fn, used_answer_key: bool)."""
    from mile_franka.tutorial import loss_exercise
    if _HAS_SOLUTION:
        b = _sample_batch()
        ref = _solution.mile_cont_loss_fn(*b)
        try:
            cand = loss_exercise.mile_cont_loss_fn(*b)
            ok = all(torch.allclose(c, r, atol=1e-6) for c, r in zip(cand, ref))
        except Exception:
            ok = False
        if ok:
            print("[tutorial] using YOUR loss implementation ✔")
            return loss_exercise.mile_cont_loss_fn, False
        print("[tutorial] your loss is missing/incorrect — applying the answer key so you can train.")
        return _solution.mile_cont_loss_fn, True
    else:
        # Solutions not distributed — participant must implement the exercise.
        print("[tutorial] using YOUR loss implementation (no answer key available).")
        return loss_exercise.mile_cont_loss_fn, False
