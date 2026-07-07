"""Resolve which loss to train with: the participant's exercise if correct, else the
reference answer key (auto-apply). Used by scripts/tutorial_train.py to monkeypatch
mile.algorithm.mile_cont_loss_fn WITHOUT editing production code.
"""
import base64
import types
import torch

# Reference solution stored as base64 to avoid distributing plain source.
_SOLUTION_B64 = (
    "IiIiUmVmZXJlbmNlIGFuc3dlciBmb3IgdGhlIGxvc3MgZXhlcmNpc2Ug4oCUIGlkZW50aWNhbCB0"
    "byBtaWxlLmFsZ29yaXRobS5taWxlX2NvbnRfbG9zc19mbi4iIiIKaW1wb3J0IHRvcmNoCmltcG9y"
    "dCB0b3JjaC5kaXN0cmlidXRpb25zIGFzIEQKaW1wb3J0IHRvcmNoLm5uLmZ1bmN0aW9uYWwgYXMg"
    "RgoKZnJvbSBzdGFibGVfYmFzZWxpbmVzMy5jb21tb24uZGlzdHJpYnV0aW9ucyBpbXBvcnQgc3Vt"
    "X2luZGVwZW5kZW50X2RpbXMKCmRldmljZSA9ICJjdWRhIiBpZiB0b3JjaC5jdWRhLmlzX2F2YWls"
    "YWJsZSgpIGVsc2UgImNwdSIKCgpkZWYgX2Rpc2NfbG9zcyhwcmVkX3Byb2JzLCBndF9sYWJlbHMs"
    "IHJlZHVjdGlvbj0ibWVhbiIpOgogICAgcHJlZF9wcm9icyA9IHRvcmNoLmNsYW1wKHByZWRfcHJv"
    "YnMsIDFlLTcsIDEgLSAxZS03KQogICAgcmV0dXJuIEYubmxsX2xvc3ModG9yY2gubG9nKHByZWRf"
    "cHJvYnMpLCBndF9sYWJlbHMsIHJlZHVjdGlvbj1yZWR1Y3Rpb24pCgoKZGVmIG1pbGVfY29udF9s"
    "b3NzX2ZuKGludGVydmVudGlvbl9wcm9iLCBtdSwgbG9nX3N0ZCwgZ3JvdW5kX3RydXRoX2FjdGlv"
    "biwKICAgICAgICAgICAgICAgICAgICAgIGdyb3VuZF90cnV0aF9pbnRlcnZlbnRpb24sIExBTUJE"
    "QTE9MS4wLCBMQU1CREEyPTEuMCwKICAgICAgICAgICAgICAgICAgICAgIHJlZHVjdGlvbj0ibWVh"
    "biIpOgogICAgZGlzY3JldGVfbG9zcyA9IF9kaXNjX2xvc3MoaW50ZXJ2ZW50aW9uX3Byb2IsIGdy"
    "b3VuZF90cnV0aF9pbnRlcnZlbnRpb24sIHJlZHVjdGlvbj1yZWR1Y3Rpb24pCiAgICBpZHggPSB0"
    "b3JjaC5sb2dpY2FsX2FuZChncm91bmRfdHJ1dGhfaW50ZXJ2ZW50aW9uID09IDEsIGludGVydmVu"
    "dGlvbl9wcm9iWzosIC0xXSA+IDAuMCkKICAgIGlmIGlkeC5zdW0oKSA9PSAwOgogICAgICAgIGNv"
    "bnRpbnVvdXNfbG9zcyA9IHRvcmNoLnRlbnNvcigwLjApLnRvKGRldmljZSkKICAgIGVsc2U6CiAg"
    "ICAgICAgZGlzdCA9IEQuTm9ybWFsKG11W2lkeF0sIGxvZ19zdGRbaWR4XS5leHAoKSkKICAgICAg"
    "ICBsb2dfcHJvYiA9IHN1bV9pbmRlcGVuZGVudF9kaW1zKGRpc3QubG9nX3Byb2IoZ3JvdW5kX3Ry"
    "dXRoX2FjdGlvbltpZHhdKSkKICAgICAgICBjb250aW51b3VzX2xvc3MgPSAtbG9nX3Byb2IubWVh"
    "bigpCiAgICBsb3NzID0gTEFNQkRBMSAqIGNvbnRpbnVvdXNfbG9zcyArIExBTUJEQTIgKiBkaXNj"
    "cmV0ZV9sb3NzCiAgICByZXR1cm4gbG9zcywgY29udGludW91c19sb3NzLCBkaXNjcmV0ZV9sb3NzCg=="
)

def _load_solution():
    mod = types.ModuleType("_loss_solution")
    exec(compile(base64.b64decode(_SOLUTION_B64), "_loss_solution", "exec"), mod.__dict__)
    return mod

_solution = _load_solution()
_HAS_SOLUTION = True


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
