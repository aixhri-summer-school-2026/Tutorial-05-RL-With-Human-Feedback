from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fixed down-facing wrist orientation as a quaternion (qx, qy, qz, qw): a 180-deg
# rotation about +X points the gripper at the table. CONFIRM@bringup: the exact quat
# the multipanda Cartesian-impedance controller expects for "pointing down".
DOWN_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Known wrist-down "neutral retract" joint configuration (panda_joint1..7, radians).
# The soft Cartesian-impedance controller cannot reliably *achieve* a straight-down wrist
# from an arbitrary start (rotational stiffness is weak and the 10 Hz moving target never
# lets orientation settle), so on the REAL robot we first drive the arm to this fixed joint
# config, THEN activate the Cartesian controller (which captures the now-down EE as its
# equilibrium); afterwards the policy/scripted loop only commands small position deltas with
# DOWN_QUAT held.
#
# This value is the hucebot `MoveToStartExampleController` q_goal (hardcoded in
# franka_example_controllers/src/comless/move_to_start_example_controller.cpp). That controller
# is defined ONLY in the real config (franka_bringup/config/real/single_controllers.yaml), so
# on real we reach this config by activating move_to_start (it ignores external goals — this
# constant just documents/mirrors where the arm lands and seeds any custom joint move). The
# multipanda SIM config has no move_to_start controller, so sim homes via the Cartesian path
# instead (see envs/registration.py). CONFIRM@bringup: that this q_goal is genuinely wrist-down
# and clears the workspace on the real FR3; refine via IK at the home TCP point if not.
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
