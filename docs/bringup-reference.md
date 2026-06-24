# FR3 / multipanda bringup reference

Hard-won facts extracted from development; keep this file, not the verbose inline comments.

## ROS topic / action names

| Thing | multipanda sim | FR3 (franka_ros2) |
|---|---|---|
| EE pose (read) | `/cartesian_impedance/cartesian_pos_curr` | same |
| EE setpoint (write) | `/cartesian_impedance/equilibrium_pose` | same |
| Controller | `custom_cartesian_impedance_controller` | same |
| Gripper grasp | `/panda_gripper_sim_node/grasp` | `/franka_gripper/grasp` |
| Gripper move (open) | same node `/move` | same node `/move` |
| Error recovery | `/error_recovery` | `/franka_control/error_recovery` |
| base frame | `panda_link0` | `fr3_link0` |

All overridable via env vars: `MILE_GRASP_ACTION`, `MILE_ERROR_RECOVERY_ACTION`, `MILE_CONTROLLER`.

> CONFIRM@bringup: verify gripper and error-recovery names against `ros2 action list` on the real graph.

## EE orientation — DOWN_QUAT

`xyzw = [1, 0, 0, 0]` → RPY [180°, 0, 0] = fingers pointing straight down.  
Previous value `[0.9239, -0.3827, 0, 0]` was wrong: it yawed the gripper −45°.  
Safe with the impedance controller; the `cartesian_pose_target_controller` has a near-singular
wrist issue at joint5 ≈ 0 with this quaternion — do not use that controller on the real FR3.

## Q_HOME — neutral retract pose

`[-0.008, -0.005, 0.011, -1.563, 0.005, 1.603, 0.850]` rad (panda_joint1..7).  
Mirrors the hucebot `MoveToStartExampleController` q_goal
(`franka_example_controllers/src/comless/move_to_start_example_controller.cpp`).
Exists only in the real controller config; the sim homes via Cartesian ramping instead.

> CONFIRM@bringup: verify Q_HOME reaches a wrist-down pose that clears the workspace.

## Real FR3 quirks

- **Do not use `move_to_start` on FR3.** Switching Effort→CartesianPose→Effort trips
  `communication_constraints_violation` in libfranka while the motion generator is still active.
  The impedance controller activates at the current orientation and `_home()` ramps orientation
  under impedance control instead.
- **`reset_controller_target_on_reset=True` on real.** Deactivating the Effort controller to
  reset its target also trips the libfranka reflex, so we re-seed the equilibrium target by
  publishing the current EE pose (no mode switch).
- **Read EE from `/cartesian_impedance/cartesian_pos_curr`, not `panda_hand`.** The hand link
  is ~0.103 m above the controller's EE frame; using it causes the scripted policy to command
  the arm underground before triggering a grasp.
- **Use the `Move` action to open the gripper** (not `Grasp`). `Grasp` only clamps inward with
  force and reports success even when fingers close; `Move` positions to a target width.
  Some sim gripper nodes lack `Move` — the backend falls back to `Grasp` for sim opening.

## Sim world-frame flip

In the multipanda MuJoCo sim the `world` frame and `panda_link0` are co-located but rotated
180° about z (link0 quat xyzw = [0,0,1,0]).  All body poses read from `get_body_state` arrive
in `world` and must be flipped (negate x, y) before use in the base frame; body-set calls flip
back.  `MultipandaRosBackend._flip_xy_pos()` implements this.

## iceoryx + CLI subprocesses

The sim's CycloneDDS iceoryx shared-memory transport breaks `ros2` CLI discovery (node list
empty → `ros2 param` fails, `ros2 control` times out) while the data path still needs iceoryx
for images.  Workaround: set `CYCLONEDDS_URI=""` for CLI subprocesses in sim only (`_run_ros2`
with `no_iceoryx=True`).  The real path is unaffected.

## cartesian_pose_target_controller substep clamping

This controller wraps each setpoint in a smoothstep over `target_duration_s` ≈ one env tick.
It is velocity-continuous only if exactly one small setpoint arrives per tick (peak acceleration
`a_peak ≈ 5.78 × hop / target_duration²` must stay under the FR3 limit).  The backend clamps
each hop to `setpoint_substep_m` and waits the full tick before publishing the next.  Flooding
substeps within one tick restarts the ramp at non-zero velocity → discontinuity reflex.
**Not the default controller on FR3** (wrist-down singularity); kept as a debug override.
