"""Keyboard teleop device — the default device for the summer-school tutorial.

Mirrors JoystickDevice (segment-toggle intervention, gripper toggle, injectable
reader) so TeleopIntervener and the collector work unchanged. The reader returns a
snapshot whose `.keys` is the set of currently-pressed key names. The default reader
lazily imports pygame (already in the image) and opens a small focus window.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

import numpy as np

from mile_franka.teleop.base import TeleopDevice, TeleopReading


class KeyboardDevice(TeleopDevice):
    """Maps keyboard input to a 4-DoF [dx, dy, dz, gripper] action.

    Default map: w/s=+x/-x, a/d=+y/-y, q/e=+z/-z, space=clutch (segment toggle),
    g=gripper toggle, enter=done, backspace=discard. Movement keys produce a fixed
    delta of translation_scale metres per step. Gripper starts as nan (passthrough)
    until the first toggle. hold_confirm_s/debounce_s filter key repeat/bounce on the
    edge-triggered keys (clutch/gripper/done/discard).
    """

    def __init__(self, translation_scale: float = 0.02,
                 key_xplus: str = "w", key_xminus: str = "s",
                 key_yplus: str = "a", key_yminus: str = "d",
                 key_zplus: str = "q", key_zminus: str = "e",
                 clutch_key: str = "space", gripper_key: str = "g",
                 done_key: str = "enter", discard_key: str = "backspace",
                 segment_mode: bool = True, gripper_toggle: bool = True,
                 hold_confirm_s: float = 0.0, debounce_s: float = 0.2,
                 reader: Optional[Callable[[], object]] = None):
        self.translation_scale = translation_scale
        self.key_xplus, self.key_xminus = key_xplus, key_xminus
        self.key_yplus, self.key_yminus = key_yplus, key_yminus
        self.key_zplus, self.key_zminus = key_zplus, key_zminus
        self.clutch_key, self.gripper_key = clutch_key, gripper_key
        self.done_key, self.discard_key = done_key, discard_key
        self.segment_mode = segment_mode
        self.gripper_toggle = gripper_toggle
        self.hold_confirm_s = hold_confirm_s
        self.debounce_s = debounce_s
        self._in_segment = False
        self._gripper_state = -1.0  # open
        self._prev_down: dict = {}
        self._press_start: dict = {}
        self._triggered_this_press: dict = {}
        self._last_trigger: dict = {}
        self._reader = reader if reader is not None else self._default_reader()

    def reset(self) -> None:
        self._in_segment = False

    def sync_gripper_state(self, closed: bool) -> None:
        self._gripper_state = 1.0 if closed else -1.0

    @staticmethod
    def _default_reader() -> Callable[[], object]:
        import pygame  # lazy: only needed for live use

        pygame.init()
        pygame.display.set_mode((240, 80))
        pygame.display.set_caption("MILE keyboard teleop — keep focused")
        names = {
            "w": pygame.K_w, "s": pygame.K_s, "a": pygame.K_a, "d": pygame.K_d,
            "q": pygame.K_q, "e": pygame.K_e, "g": pygame.K_g,
            "space": pygame.K_SPACE, "enter": pygame.K_RETURN,
            "backspace": pygame.K_BACKSPACE,
        }

        class _Snapshot:
            __slots__ = ("keys",)

        def _read() -> object:
            pygame.event.pump()
            pressed = pygame.key.get_pressed()
            snap = _Snapshot()
            snap.keys = {n for n, code in names.items() if pressed[code]}
            return snap

        return _read

    def _confirmed_edge(self, keys: Iterable[str], name: str, now: float) -> bool:
        cur = name in keys
        prev = self._prev_down.get(name, False)
        self._prev_down[name] = cur
        if not prev and cur:
            self._press_start[name] = now
            self._triggered_this_press[name] = False
        if prev and not cur:
            self._press_start.pop(name, None)
        if not cur or self._triggered_this_press.get(name, True):
            return False
        held = now - self._press_start.get(name, now)
        since = now - self._last_trigger.get(name, -999.0)
        if held >= self.hold_confirm_s and since >= self.debounce_s:
            self._last_trigger[name] = now
            self._triggered_this_press[name] = True
            return True
        return False

    def read(self) -> TeleopReading:
        now = time.monotonic()
        keys = set(getattr(self._reader(), "keys", ()))

        dx = (self.key_xplus in keys) - (self.key_xminus in keys)
        dy = (self.key_yplus in keys) - (self.key_yminus in keys)
        dz = (self.key_zplus in keys) - (self.key_zminus in keys)
        action_xyz = np.array([dx, dy, dz], dtype=np.float32) * self.translation_scale

        clutch_edge = self._confirmed_edge(keys, self.clutch_key, now)
        gripper_edge = self._confirmed_edge(keys, self.gripper_key, now)
        done = self._confirmed_edge(keys, self.done_key, now)
        discard = self._confirmed_edge(keys, self.discard_key, now)

        if discard:
            self._in_segment = False

        if self.segment_mode:
            if clutch_edge:
                self._in_segment = not self._in_segment
            intervene = self._in_segment
        else:
            intervene = (self.clutch_key in keys) or bool(np.any(action_xyz != 0.0))

        if self.gripper_toggle:
            if gripper_edge:
                self._gripper_state = 1.0 if self._gripper_state < 0 else -1.0
            gripper = self._gripper_state if self._gripper_toggled() else float("nan")
        else:
            gripper = 1.0 if self.gripper_key in keys else -1.0

        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], gripper],
                          dtype=np.float32)
        return TeleopReading(action=action, intervene=bool(intervene),
                             done=bool(done), discard=bool(discard))

    def _gripper_toggled(self) -> bool:
        return self.gripper_key in self._last_trigger
