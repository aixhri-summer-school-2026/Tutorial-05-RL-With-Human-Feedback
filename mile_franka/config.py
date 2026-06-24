from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# EE pointing straight down: xyzw = [1,0,0,0] = RPY[180°, 0, 0]. See docs/bringup-reference.md.
MULTIPANDA_DOWN_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
FR3_DOWN_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DOWN_QUAT: np.ndarray = MULTIPANDA_DOWN_QUAT

# Neutral wrist-down joint config (panda_joint1..7, rad). Mirrors hucebot MoveToStartExampleController.
# Real backend reaches this via move_to_start; sim homes via Cartesian ramp. See docs/bringup-reference.md.
Q_HOME: np.ndarray = np.array(
    [-0.008, -0.005, 0.011, -1.563, 0.005, 1.603, 0.850], dtype=np.float32)


@dataclass
class StackTaskConfig:
    """Numbers shared across sim and real for the cube-stacking task.

    Lengths are meters in the robot base frame.
    """

    cube_size: float = 0.05                  # cube edge length (< gripper max open)
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

    # Success: top cube remains at stack height after release. XY is not part of the final
    # predicate for sim because a physically stable cube at the target height is enough.
    success_xy_tol: float = np.inf           # retained for diagnostics / optional tightening
    success_z_tol: float = 0.01              # max error in stacked height (m)
    success_stable_steps: int = 10           # post-release steps at stack height before success
    success_pos_stable_tol: float = 0.002    # max cube-center motion per stable step (m)

    # Reset randomization: cube centers drawn from an inner margin of the workspace.
    reset_margin: float = 0.05               # m kept clear of workspace xy edges
    reset_min_separation: float = 0.16       # m minimum xy gap between the two cubes
