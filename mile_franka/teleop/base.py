"""Device-agnostic teleoperation interface.

A TeleopDevice converts raw human input into a 4-DoF action and the two control
signals MILE needs: whether the human is intervening, and whether they ended the
episode. SpaceMouse (home lab) and Vive (France) both implement this interface, so
the data collector is identical across labs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TeleopReading:
    """One poll of a teleop device.

    action: float32 array (4,) = [dx, dy, dz, gripper]; gripper in {-1 open, +1 close}.
    intervene: True when the human is currently taking over control (maps to MILE's nu).
    done: True when the human signals the episode should end.
    """

    action: np.ndarray
    intervene: bool
    done: bool


class TeleopDevice(ABC):
    """Abstract teleop device. Implementations poll hardware and return a reading."""

    @abstractmethod
    def read(self) -> TeleopReading:
        """Return the current reading. Must be non-blocking / fast (called per step)."""

    def close(self) -> None:
        """Release any hardware handles. Default is a no-op."""
