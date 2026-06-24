import numpy as np
from mile_franka.teleop.keyboard import KeyboardDevice


def _dev(keys_seq):
    """KeyboardDevice whose reader yields successive frozensets of pressed keys."""
    frames = list(keys_seq)
    state = {"i": 0}

    def reader():
        i = min(state["i"], len(frames) - 1)
        state["i"] += 1

        class S:
            keys = frames[i]
        return S()

    return KeyboardDevice(reader=reader, translation_scale=0.02,
                          hold_confirm_s=0.0, debounce_s=0.0)


def test_movement_keys_map_to_xyz_delta():
    dev = _dev([{"w"}])
    r = dev.read()
    assert r.action[0] > 0 and r.action[1] == 0 and r.action[2] == 0
    assert abs(r.action[0] - 0.02) < 1e-6


def test_opposite_keys_have_opposite_sign():
    assert _dev([{"w"}]).read().action[0] > 0
    assert _dev([{"s"}]).read().action[0] < 0
    assert _dev([{"a"}]).read().action[1] > 0
    assert _dev([{"d"}]).read().action[1] < 0


def test_clutch_space_toggles_intervention_segment():
    dev = _dev([set(), {"space"}, set(), {"space"}, set()])
    assert dev.read().intervene is False      # idle
    assert dev.read().intervene is True       # space pressed -> enter segment
    assert dev.read().intervene is True       # still in segment
    assert dev.read().intervene is False      # space again -> exit segment
    assert dev.read().intervene is False


def test_gripper_toggle_alternates_and_starts_nan():
    dev = _dev([set(), {"g"}, set(), {"g"}, set()])
    assert np.isnan(dev.read().action[3])     # before first toggle: nan passthrough
    assert dev.read().action[3] == 1.0        # g -> close
    assert dev.read().action[3] == 1.0        # held value persists
    assert dev.read().action[3] == -1.0       # g -> open


def test_done_and_discard_keys():
    assert _dev([{"enter"}]).read().done is True
    assert _dev([{"backspace"}]).read().discard is True


def test_reset_clears_segment():
    dev = _dev([{"space"}, set()])
    assert dev.read().intervene is True
    dev.reset()
    assert dev._in_segment is False


def test_sync_gripper_state():
    dev = _dev([set()])
    dev.sync_gripper_state(closed=True)
    assert dev._gripper_state == 1.0
