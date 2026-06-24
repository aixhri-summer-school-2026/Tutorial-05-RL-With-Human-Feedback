from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fixed down-facing wrist orientations as quaternions (qx, qy, qz, qw).
# Both stacks command the EE "straight down" as a 180-deg rotation about base +X
# (xyzw = [1,0,0,0]); the publish path maps the array straight onto
# geometry_msgs/Quaternion (x,y,z,w). VERIFIED on the live FR3 2026-06-19 via
# tf2_echo fr3_link0->fr3_hand_tcp: [1,0,0,0] -> RPY[180,0,0] (fingers axis-aligned).
# The previous FR3 value [0.9239,-0.3827,0,0] was a mis-derived "correction" that
# actually yawed the gripper -45 deg (RPY[180,0,-45]). If the lab needs a deliberate
# grasp yaw, set xyzw for RPY[180,0,+/-yaw] (e.g. +45 -> [0.9239,0.3827,0,0]).
MULTIPANDA_DOWN_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
# RESOLVED 2026-06-20: clean axis-aligned down. The old value [0.9239,-0.3827,0,0]
# (RPY[180,0,-45]) yawed the gripper -45deg at home -- the longstanding "gripper off by 45deg"
# bug. The historical worry was that commanding [1,0,0,0] near the home pose lands joint5~=0
# (near-singular wrist) and a CartesianPose motion generator would throw
# cartesian_motion_generator_joint_velocity_discontinuity. That worry is specific to the
# CartesianPose controller; the soft Cartesian-IMPEDANCE controller (Effort/torque, the path
# we use) has no motion generator. VERIFIED on hardware: from the -46.5deg as-homed yaw,
# commanding [1,0,0,0] rotated the wrist to -3.4deg yaw with 0.22cm position drift and NO
# reflex (scripts/impedance_orient_test.py). The ~3deg residual is rotational steady-state lag
# under rot_stiff=25; raise rot_stiff or allow more settle to tighten it.
FR3_DOWN_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DOWN_QUAT: np.ndarray = MULTIPANDA_DOWN_QUAT

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
# and clears the workspace on the real Franka; refine via IK at the home TCP point if not.
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
