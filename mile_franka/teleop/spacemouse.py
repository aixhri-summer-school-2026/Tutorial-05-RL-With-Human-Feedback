"""SpaceMouse teleop device for the home-lab development setup.

The raw-reader is injectable so the mapping logic runs without hardware. By default it
uses the `pyspacemouse` library; the import is lazy so the module loads on machines
without the device.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mile_franka.teleop.base import TeleopDevice, TeleopReading


class SpaceMouseDevice(TeleopDevice):
    """Maps a 3Dconnexion SpaceMouse to a 4-DoF [dx, dy, dz, gripper] action.

    Args:
        translation_scale: meters of EE delta per unit of puck deflection.
        deadband: puck deflection below this (abs) is treated as zero.
        reader: zero-arg callable returning an object with .x, .y, .z, .buttons.
                Defaults to a pyspacemouse-backed reader.
    """

    def __init__(self, translation_scale: float = 0.02, deadband: float = 0.1,
                 reader: Optional[Callable[[], object]] = None):
        self.translation_scale = translation_scale
        self.deadband = deadband
        self._reader = reader if reader is not None else self._default_reader()

    @staticmethod
    def _default_reader() -> Callable[[], object]:
        import pyspacemouse  # lazy: only needed with real hardware

        pyspacemouse.open()
        return pyspacemouse.read

    def read(self) -> TeleopReading:
        s = self._reader()
        trans = np.array([s.x, s.y, s.z], dtype=np.float32)
        trans[np.abs(trans) < self.deadband] = 0.0
        action_xyz = trans * self.translation_scale

        buttons = tuple(getattr(s, "buttons", (0, 0)))
        gripper_close = bool(buttons[0]) if len(buttons) > 0 else False
        done = bool(buttons[1]) if len(buttons) > 1 else False

        gripper = 1.0 if gripper_close else -1.0
        moved = bool(np.any(action_xyz != 0.0))
        intervene = moved or gripper_close

        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], gripper],
                          dtype=np.float32)
        return TeleopReading(action=action, intervene=intervene, done=done)

    def close(self) -> None:
        try:
            import pyspacemouse

            pyspacemouse.close()
        except Exception:
            pass
