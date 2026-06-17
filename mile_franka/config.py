from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fixed down-facing wrist orientation as a quaternion (qx, qy, qz, qw): a 180-deg
# rotation about +X points the gripper at the table. CONFIRM@bringup: the exact quat
# the multipanda Cartesian-impedance controller expects for "pointing down".
DOWN_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class StackTaskConfig:
    """Numbers shared across sim and real for the cube-stacking task.

    Lengths are meters in the robot base frame.
    """

    cube_size: float = 0.04                  # cube edge length (< gripper max open)
    gripper_open_width: float = 0.08         # finger separation when open (m)
    gripper_closed_width: float = 0.0        # finger separation when fully closed (m)

    # Axis-aligned workspace box the EE target is clipped into: [low, high] per axis.
    workspace_low: np.ndarray = field(
        default_factory=lambda: np.array([0.30, -0.30, 0.02], dtype=np.float32))
    workspace_high: np.ndarray = field(
        default_factory=lambda: np.array([0.70, 0.30, 0.40], dtype=np.float32))
    table_z: float = 0.02                    # z of the table surface (cube resting plane)

    action_scale: float = 0.055              # meters of EE delta per 10 Hz action tick
    max_steps: int = 150                     # truncation horizon

    # Success: top cube centered on bottom cube and released.
    success_xy_tol: float = 0.02             # max horizontal center offset (m)
    success_z_tol: float = 0.01              # max error in stacked height (m)

    # Reset randomization: cube centers drawn from an inner margin of the workspace.
    reset_margin: float = 0.05               # m kept clear of workspace xy edges
    reset_min_separation: float = 0.16       # m minimum xy gap between the two cubes
