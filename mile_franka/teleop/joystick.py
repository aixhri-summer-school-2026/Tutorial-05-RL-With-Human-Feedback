"""Gamepad (Xbox/PS) teleop device — the device the original MILE paper used.

Mirrors SpaceMouseDevice: the raw reader is injectable so the mapping logic runs without
hardware. The default reader lazily imports pygame (already in the image), so the module
loads on machines without a controller.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from mile_franka.teleop.base import TeleopDevice, TeleopReading


class JoystickDevice(TeleopDevice):
    """Maps a gamepad to a 4-DoF [dx, dy, dz, gripper] action.

    Segment mode (default): clutch_button is a toggle — press once to enter an
    intervention segment, press again to exit. intervene stays True for the whole
    segment so the operator can plan between frames without constant button pressure.
    Outside a segment the robot runs under the policy and nothing is recorded.

    Legacy mode (segment_mode=False): intervene is True only while the clutch is
    held OR a stick is moved OR the gripper button is pressed (original per-frame
    clutch semantics).

    Gripper: button press toggles between open (-1) and close (+1). Before the first
    explicit toggle in a segment, action[3]=nan signals TeleopIntervener to pass
    through the policy's gripper command unchanged.

    Button filtering (ghost + bounce protection):
        A button press is accepted only after it has been held continuously for at
        least hold_confirm_s (ghost filter). After an accepted trigger, the button is
        locked out for debounce_s (bounce filter). Together these reject sub-50ms
        phantom spikes and rapid mechanical bounce.

    Args:
        translation_scale: metres of EE delta per unit of stick deflection.
        deadband: stick deflection below this (abs) is treated as zero.
        ax_x, ax_y, ax_z: axis indices mapped to dx, dy, dz.
        ax_x_sign, ax_y_sign, ax_z_sign: invert an axis with -1.0.
        clutch_button: toggle-segment button (segment_mode) or hold-to-intervene.
        gripper_button: rising-edge toggle between open/close.
        done_button: signals episode end.
        segment_mode: True = stateful segment toggle; False = per-frame clutch.
        gripper_toggle: True = toggle on press; False = hold-to-close.
        hold_confirm_s: minimum hold duration to accept a press (ghost filter).
        debounce_s: lockout after an accepted trigger (bounce filter).
        reader: zero-arg callable returning an object with .axes and .buttons.
    """

    def __init__(self, translation_scale: float = 0.02, deadband: float = 0.1,
                 ax_x: int = 0, ax_y: int = 1, ax_z: int = 4,
                 ax_x_sign: float = 1.0, ax_y_sign: float = 1.0, ax_z_sign: float = 1.0,
                 clutch_button: int = 5, gripper_button: int = 0, done_button: int = 7,
                 discard_button: int = 6,
                 segment_mode: bool = True,
                 gripper_toggle: bool = True,
                 hold_confirm_s: float = 0.2,
                 debounce_s: float = 0.3,
                 reader: Optional[Callable[[], object]] = None):
        self.translation_scale = translation_scale
        self.deadband = deadband
        self.ax_x, self.ax_y, self.ax_z = ax_x, ax_y, ax_z
        self.ax_x_sign, self.ax_y_sign, self.ax_z_sign = ax_x_sign, ax_y_sign, ax_z_sign
        self.clutch_button = clutch_button
        self.gripper_button = gripper_button
        self.done_button = done_button
        self.discard_button = discard_button
        self.segment_mode = segment_mode
        self.gripper_toggle = gripper_toggle
        self.hold_confirm_s = hold_confirm_s
        self.debounce_s = debounce_s
        self._in_segment: bool = False
        self._gripper_state: float = -1.0  # open
        # Per-button state for filtering
        self._prev_buttons: Dict[int, bool] = {}
        self._press_start: Dict[int, float] = {}    # time button first went down
        self._triggered_this_press: Dict[int, bool] = {}  # fired once per physical press
        self._last_trigger: Dict[int, float] = {}   # time of last accepted trigger
        self._reader = reader if reader is not None else self._default_reader()

    def reset(self) -> None:
        """Reset stateful control flags on episode reset."""
        self._in_segment = False

    def sync_gripper_state(self, closed: bool) -> None:
        """Sync gripper state from the robot on segment entry (called by TeleopIntervener)."""
        self._gripper_state = 1.0 if closed else -1.0

    @staticmethod
    def _default_reader() -> Callable[[], object]:
        import pygame  # lazy: only needed with real hardware

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick found — check /dev/input mapping and that a pad is connected")
        js = pygame.joystick.Joystick(0)
        js.init()

        class _Snapshot:
            __slots__ = ("axes", "buttons")

        def _read() -> object:
            pygame.event.pump()
            snap = _Snapshot()
            snap.axes = [js.get_axis(i) for i in range(js.get_numaxes())]
            snap.buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
            return snap

        return _read

    @staticmethod
    def _axis(axes: Sequence[float], idx: int) -> float:
        return float(axes[idx]) if 0 <= idx < len(axes) else 0.0

    @staticmethod
    def _button(buttons: Sequence[int], idx: int) -> bool:
        return bool(buttons[idx]) if 0 <= idx < len(buttons) else False

    def _confirmed_edge(self, buttons: Sequence[int], idx: int, now: float) -> bool:
        """True once per physical press, after hold_confirm_s hold and debounce_s lockout.

        Ghost filter: a press that is released before hold_confirm_s is silently dropped.
        Bounce filter: after an accepted trigger, the button is locked for debounce_s.
        Only one trigger fires per continuous press (released→re-pressed resets the timer).
        """
        cur = self._button(buttons, idx)
        prev = self._prev_buttons.get(idx, False)
        self._prev_buttons[idx] = cur

        if not prev and cur:
            # Physical rising edge: start hold timer, clear fired flag
            self._press_start[idx] = now
            self._triggered_this_press[idx] = False

        if prev and not cur:
            # Released: clear hold timer
            self._press_start.pop(idx, None)

        if not cur or self._triggered_this_press.get(idx, True):
            return False  # not held, or already fired this press

        held_s = now - self._press_start.get(idx, now)
        since_last = now - self._last_trigger.get(idx, -999.0)

        if held_s >= self.hold_confirm_s and since_last >= self.debounce_s:
            self._last_trigger[idx] = now
            self._triggered_this_press[idx] = True  # one trigger per press
            return True

        return False

    def read(self) -> TeleopReading:
        now = time.monotonic()
        s = self._reader()
        axes = getattr(s, "axes", ())
        buttons = getattr(s, "buttons", ())

        trans = np.array([self._axis(axes, self.ax_x) * self.ax_x_sign,
                          self._axis(axes, self.ax_y) * self.ax_y_sign,
                          self._axis(axes, self.ax_z) * self.ax_z_sign], dtype=np.float32)
        trans[np.abs(trans) < self.deadband] = 0.0
        action_xyz = trans * self.translation_scale

        clutch_edge = self._confirmed_edge(buttons, self.clutch_button, now)
        gripper_edge = self._confirmed_edge(buttons, self.gripper_button, now)
        done = self._confirmed_edge(buttons, self.done_button, now)
        discard = self._confirmed_edge(buttons, self.discard_button, now)
        clutch_held = self._button(buttons, self.clutch_button)

        if discard:
            self._in_segment = False  # exit any active segment on discard

        # --- intervention state ---
        if self.segment_mode:
            if clutch_edge:
                self._in_segment = not self._in_segment
            intervene = self._in_segment
        else:
            moved = bool(np.any(action_xyz != 0.0))
            intervene = clutch_held or moved or self._button(buttons, self.gripper_button)

        # --- gripper ---
        if self.gripper_toggle:
            if gripper_edge:
                self._gripper_state = 1.0 if self._gripper_state < 0 else -1.0
            gripper = self._gripper_state
        else:
            gripper = 1.0 if self._button(buttons, self.gripper_button) else -1.0

        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], gripper],
                          dtype=np.float32)
        return TeleopReading(action=action, intervene=intervene, done=done, discard=discard)

    def close(self) -> None:
        try:
            import pygame

            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
