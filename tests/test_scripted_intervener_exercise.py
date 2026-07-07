"""Tests for the scripted intervener exercise.

We test three behavioural properties that any reasonable implementation must satisfy:
1. Large translation drift → intervene.
2. Identical actions → don't intervene.
3. Gripper disagreement alone → intervene.

The reference solution is also tested directly to confirm it satisfies all three.
"""
import numpy as np
import pytest


def _ref():
    from mile_franka.tutorial._scripted_intervener_solution import should_intervene
    return should_intervene


def _candidate():
    from mile_franka.tutorial.scripted_intervener_exercise import should_intervene
    return should_intervene


def _skip_if_not_implemented(fn):
    import inspect
    src = inspect.getsource(fn)
    if "raise NotImplementedError" in src:
        pytest.skip("exercise not yet implemented (blank scaffold)")


# ── reference solution ───────────────────────────────────────────────────────

def test_ref_large_drift_triggers():
    fn = _ref()
    expert = np.array([1.0, 0.0, 0.0, 0.0])
    rollout = np.array([0.0, 0.0, 0.0, 0.0])
    assert fn(expert, rollout) == True

def test_ref_no_drift_no_trigger():
    fn = _ref()
    action = np.array([0.1, 0.2, 0.0, 1.0])
    assert fn(action, action.copy()) == False

def test_ref_gripper_disagree_triggers():
    fn = _ref()
    # Close to same position, but gripper open vs closed
    expert = np.array([0.0, 0.0, 0.0, 1.0])   # gripper closed
    rollout = np.array([0.01, 0.0, 0.0, -1.0]) # gripper open
    assert fn(expert, rollout) == True


# ── candidate exercise ───────────────────────────────────────────────────────

def test_candidate_large_drift_triggers():
    fn = _candidate()
    _skip_if_not_implemented(fn)
    expert = np.array([1.0, 0.0, 0.0, 0.0])
    rollout = np.array([0.0, 0.0, 0.0, 0.0])
    assert fn(expert, rollout) == True, "large translation drift should trigger intervention"

def test_candidate_no_drift_no_trigger():
    fn = _candidate()
    _skip_if_not_implemented(fn)
    action = np.array([0.1, 0.2, 0.0, 1.0])
    assert fn(action, action.copy()) == False, "identical actions should not trigger intervention"

def test_candidate_gripper_disagree_triggers():
    fn = _candidate()
    _skip_if_not_implemented(fn)
    expert = np.array([0.0, 0.0, 0.0, 1.0])
    rollout = np.array([0.01, 0.0, 0.0, -1.0])
    assert fn(expert, rollout) == True, "gripper disagreement should trigger intervention"

def test_candidate_returns_bool():
    fn = _candidate()
    _skip_if_not_implemented(fn)
    result = fn(np.zeros(4), np.zeros(4))
    assert isinstance(result, (bool, np.bool_)), f"should_intervene must return bool, got {type(result)}"
