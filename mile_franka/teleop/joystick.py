"""Gamepad (Xbox/PS) teleop device — the device the original MILE paper used.

Mirrors SpaceMouseDevice: the raw reader is injectable so the mapping logic runs without
hardware. The default reader lazily imports pygame (already in the image), so the module
loads on machines without a controller.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from mile_franka.teleop.base import TeleopDevice, TeleopReading


class JoystickDevice(TeleopDevice):
    """Maps a gamepad to a 4-DoF [dx, dy, dz, gripper] action with clutch semantics.

    ν (intervene) is True when the clutch is held OR any mapped axis exceeds the deadband
    OR the gripper button is held — a superset of SpaceMouse's `moved or gripper_close`.

    Args:
        translation_scale: meters of EE delta per unit of stick deflection.
        deadband: stick deflection below this (abs) is treated as zero.
        ax_x, ax_y, ax_z: axis indices mapped to dx, dy, dz.
        clutch_button, gripper_button, done_button: button indices.
        reader: zero-arg callable returning an object with `.axes` and `.buttons` sequences.
                Defaults to a pygame-backed reader.
    """

    def __init__(self, translation_scale: float = 0.02, deadband: float = 0.1,
                 ax_x: int = 0, ax_y: int = 1, ax_z: int = 4,  # Xbox: left stick x/y, right stick vertical
                 clutch_button: int = 5, gripper_button: int = 0, done_button: int = 7,  # Xbox: RB, A, Start
                 reader: Optional[Callable[[], object]] = None):
        self.translation_scale = translation_scale
        self.deadband = deadband
        self.ax_x, self.ax_y, self.ax_z = ax_x, ax_y, ax_z
        self.clutch_button = clutch_button
        self.gripper_button = gripper_button
        self.done_button = done_button
        self._reader = reader if reader is not None else self._default_reader()

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

    def read(self) -> TeleopReading:
        s = self._reader()
        axes = getattr(s, "axes", ())
        buttons = getattr(s, "buttons", ())

        trans = np.array([self._axis(axes, self.ax_x),
                          self._axis(axes, self.ax_y),
                          self._axis(axes, self.ax_z)], dtype=np.float32)
        trans[np.abs(trans) < self.deadband] = 0.0
        action_xyz = trans * self.translation_scale

        clutch = self._button(buttons, self.clutch_button)
        gripper_close = self._button(buttons, self.gripper_button)
        done = self._button(buttons, self.done_button)

        gripper = 1.0 if gripper_close else -1.0
        moved = bool(np.any(action_xyz != 0.0))
        intervene = clutch or moved or gripper_close  # ν = clutch OR motion OR gripper

        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], gripper],
                          dtype=np.float32)
        return TeleopReading(action=action, intervene=intervene, done=done)

    def close(self) -> None:
        try:
            import pygame

            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
