"""Resolve which scripted intervener decision function to use."""

try:
    from mile_franka.tutorial import _scripted_intervener_solution as _solution
    _HAS_SOLUTION = True
except ImportError:
    _solution = None
    _HAS_SOLUTION = False


def _is_implemented(fn):
    import inspect
    try:
        src = inspect.getsource(fn)
        return "raise NotImplementedError" not in src
    except Exception:
        return False


def resolve_should_intervene_fn():
    """Return (should_intervene_fn, used_answer_key: bool)."""
    from mile_franka.tutorial import scripted_intervener_exercise as exercise

    if _is_implemented(exercise.should_intervene):
        print("[tutorial] using YOUR scripted intervener ✔")
        return exercise.should_intervene, False

    if _HAS_SOLUTION:
        print("[tutorial] scripted intervener not implemented — using the reference rule.")
        return _solution.should_intervene, True

    print("[tutorial] scripted intervener not implemented (no answer key available — implement the exercise).")
    return exercise.should_intervene, False
