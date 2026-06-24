# FR3 Impedance Controller Bringup Log (2026-06-19)

## Goal

Make the real FR3 move smoothly under Cartesian impedance control and run `make eval-real`.

## RESOLUTION (2026-06-20) — impedance controller now holds AND moves

**Root cause (telemetry-confirmed, not guessed):** the franka_ros2 port **dropped the
reference's projected nullspace term**, which carried the only damping on the redundant
DOF. For a 7-DOF arm holding a 6-DOF Cartesian pose the **wrist (joint 7)** lives in the
task nullspace and gets almost no damping from `J^T D (J dq)`. Live telemetry of
`/cartesian_impedance/joint_state` on activation showed j7 `dq` growing 0.7 → **5.23 rad/s**
(high-frequency self-oscillation, felt as the hard vibration) until `joint_velocity_violation`
tripped. The earlier `damping_floor_rotational=80` "fix" was the opposite failure mode:
that much rotational Cartesian damping over-damps the tiny wrist inertia
(`D·dt/I` → discrete-instability limit) — so there is **no** good Cartesian-damping value.
The wrist needs **joint-space** damping. (The two prior symptoms — "can't move >1 cm" and
"rotational damping → reflex" — were both the pathological `damping_floor_*` values of
800/80, which are 10–20× the multipanda reference's exact critical damping of 77.5/4.)

**Fix:**
1. `mile_franka_controllers/.../custom_cartesian_impedance_controller.{cpp,hpp}`: added
   direct diagonal **joint-space velocity damping** `-joint_damping_ .* dq` (RT-safe: no
   SVD, no allocation — unlike the reference's pseudo-inverse) replacing the dropped
   nullspace damping, plus an absolute FR3 per-joint torque clamp (`[87,87,87,87,12,12,12]`)
   to stop tau wind-up (had seen a 633 Nm runaway). New YAML params
   `joint_damp_q1_to_4` (8.0) / `joint_damp_q5_to_7` (4.0); the tunable gains are now
   re-read in `on_activate` so they retune live via `ros2 param set` + deactivate/activate.
2. `config/mile_controllers.yaml`: cleaned the half-edited duplicate-key block; set the
   reference-derived config — `pos_stiff=1500`, `rot_stiff=25`, `damping_floor_*=0`
   (exact critical damping), `joint_damp 8/4`.

**Verified on hardware (operator at e-stop, 2026-06-20), live telemetry:**
- HOLD 8 s: worst `|dq|` = **0.00 rad/s**, tracking err 0.01–0.02 cm, τ ~0.1 Nm, 0 spikes.
- 2 cm +x nudge: moved 1.64 cm, peak `|dq|` 0.14 rad/s, 0 spikes.
- 2.5 cm square (4 segments, all axes): each moved 2.0–2.9 cm, peak `|dq|` 0.20 rad/s, 0 spikes.
- Steady-state lag ~0.4 cm (impedance spring offset; raise `pos_stiff` or enable
  `translational_Ki` if tighter tracking is needed).

Diagnostic tools added (host repo): `scripts/impedance_telemetry.py` (live dq/τ/err from
the controller's own debug topics, flags dq spikes) and `scripts/impedance_nudge.py`
(non-interactive single-step / square motion test). To rebuild after a .cpp change:
`docker exec franka_ros2_humble bash -lc 'source /opt/ros/humble/setup.bash && cd /ros2_ws && colcon build --packages-select mile_franka_controllers'` then relaunch `make franka-up`.

**Tuning (2026-06-20):** installed-YAML defaults now `pos_stiff=2000`, `rot_stiff=50`,
`jd 8/4`. Steady-state position lag: home settles to **~1.7 mm**, a 2 cm nudge to ~3 mm; the
"0.9 cm" seen in the smoke-test square is **dynamic** lag (2 s segments, arm still chasing),
NOT steady-state — the real 10 Hz task settles near the ~2-3 mm floor. The floor is
**friction-dominated** (~6 N, `lag ≈ F_fric/K`); do NOT crank K to chase it (kills compliance).
**Integral is the wrong tool** here and is currently dead code anyway —
`equilibriumPoseCallback` zeroes `error_i_` on **every** streamed setpoint so it never
accumulates; enabling it would need a reset-only-on-large-jump guard + a stiction deadband
(integral vs static friction limit-cycles). Decision (operator): keep ~2-3 mm + compliance,
no integral.

**Gripper 45deg bug — FIXED (2026-06-20).** `FR3_DOWN_QUAT` was `[0.9239,-0.3827,0,0]`
(RPY[180,0,-45]) → gripper yawed -45deg at home. The historical "can't command clean-down
[1,0,0,0], near-singular wrist → cartesian_motion_generator_joint_velocity_discontinuity"
worry is specific to the **CartesianPose** controller; the soft impedance controller has no
motion generator. VERIFIED (`scripts/impedance_orient_test.py`): commanding `[1,0,0,0]`
rotated the wrist -46.5deg → -3.4deg yaw, no reflex, 0.22 cm drift; with `rot_stiff=50`
the residual tightens to **-1.6deg**. Fix: `FR3_DOWN_QUAT = [1,0,0,0]` in `mile_franka/config.py`
(feeds both the smoke test and `registration.py:146`).

**Operational gotchas (cost real time — read before re-tuning on hardware):**
- **One control mode per bringup session.** The FR3 firmware rejects live Effort↔CartesianPose
  switches (`prepare command mode switch was rejected`). Whichever of
  `custom_cartesian_impedance_controller` (Effort) or `cartesian_pose_target_controller`
  (CartesianPose) activates **first** owns the mode; the other then cannot activate, and the
  first often cannot even deactivate. `make real-home-smoke` defaults to
  `MILE_REAL_STACK=fr3` → `cartesian_pose_target_controller`; to use the (fixed) impedance
  controller, pass `MILE_CONTROLLER=custom_cartesian_impedance_controller`, and if the pose
  controller was already activated this session, **restart the bringup** to clear the mode.
- **Don't churn activate/deactivate/param-set on the live controller** — rapid Effort-mode
  switches missed RT deadlines and tripped `communication_constraints_violation` (a comms/RT
  reflex, NOT an impedance instability). Single clean activation + streamed setpoints is
  fine (8 s hold + square + nudges never tripped it). To re-tune: set the param while
  **inactive**, then a single activate; or edit YAML + `colcon build` + restart bringup.
- **Even ONE deactivate→reactivate of the live Effort controller trips
  `communication_constraints_violation`** on this FR3 (then `initializeTorqueInterface` fails
  with "Cannot perform this operation while another control or read operation is running",
  and the aborted libfranka loop will NOT restart on error-recovery alone — needs a fresh
  bringup). The backend's `reset(reset_controller_target_on_reset=True)` does exactly this
  deactivate→reactivate when the controller is **already active**, so a second smoke-test run
  in the same session homed trivially but the square hit a dead loop. **FIXED 2026-06-20**:
  `ros_backend.py` `reset()` now calls `_reset_controller_target()`, which on the real FR3
  re-seeds the equilibrium target to the CURRENT pose (a plain publish, no mode switch) when
  the controller is already active, instead of cycling inactive→active; inactive controllers
  are left for the upcoming activation to capture; sim keeps the old cycle. VERIFIED: two
  back-to-back full `real-home-smoke` runs (run 2 starting from an active controller) both
  home + trace the square with **0 `communication_constraints`**. This also unblocks the
  per-episode resets in `eval-real`/`mile-real`. (Residual follow-up: a reflex that occurs
  mid-run still needs a real ErrorRecovery server + loop restart to recover without a full
  bringup; the smoke test passes no `error_recovery_action` so `_recover_from_errors` no-ops.)

**VERIFIED WORKING (2026-06-20):** from a fresh bringup,
`make real-home-smoke MILE_CONTROLLER=custom_cartesian_impedance_controller` (= `MILE_REAL_STACK=fr3`)
homes to [0.451, 0.000, 0.331] and traces the full 2.5 cm square (all 4 segments move,
~0.9 cm/segment tracking over the 2 s segment), **zero reflexes**, at the tuned K=2000.
- **`ros2 param set` needs a float literal** (`2000.0`, not `2000`): the params are declared
  `double`, an integer literal makes `get_parameter(...).as_double()` throw
  `expected [double] got [integer]` → on_activate/on_configure ERROR → controller drops to
  `unconfigured`.
- **`set_controller_state` wants the bare name** (`custom_cartesian_impedance_controller`),
  **`ros2 param`** wants the slashed node name (`/custom_cartesian_impedance_controller`).
- The launch loads the **installed** YAML (`/ros2_ws/install/.../config/`), not the src;
  editing src needs `colcon build` to take effect on the next bringup.

**Op note:** after a container restart (or bringup kill/relaunch) the `ros2` CLI **daemon**
goes stale (`xmlrpc Fault ... !rclpy.ok()`); fix with `ros2 daemon stop && ros2 daemon start`.
This is a CLI-daemon issue, not a bringup failure. To restart just the controller stack
without touching the container/daemon: `pkill -INT -f mile_bringup.launch.py` then relaunch
`make franka-up` (or the ros2 launch directly).

The "What Doesn't Work" section below is the pre-fix state, kept for history.

## System Info

- **Robot**: Franka FR3, system image 5.9.1, FCI 1.0.0
- **libfranka**: 0.20.4 (compatible with system >= 5.9.0 per [compatibility matrix](https://frankarobotics.github.io/docs/doc/libfranka/docs/compatibility_matrix.html))
- **Gripper**: attached (~0.7-1.0 kg — not in the dynamics model)
- **ROS2**: Humble, CycloneDDS, host networking
- **Kernel**: 6.8.1-1015-realtime, CPU governor=performance
- **Containers**: `franka_ros2_humble` (controller stack) + `mile_sim` (MILE Python)
- **CPU isolation**: franka on CPUs 0-3, mile_sim on 4-19

## Key Bugs Fixed

### 1. Dangling Pointer in Model Reads (CRITICAL)

**File**: `mile_franka_controllers/src/custom_cartesian_impedance_controller.cpp`

FrankaRobotModel's `getPoseMatrix()`, `getMassMatrix()`, `getCoriolisForceVector()`, `getZeroJacobian()` all return `std::array` **by value**. The original code (both ours and multipanda's) called `.data()` on the temporary and wrapped it in `Eigen::Map` — creating a dangling pointer destroyed at the semicolon. This caused:

- Random position/orientation readings → wrong error computation
- Random Jacobian matrices → wrong torque direction
- Random coriolis vectors → bias torque → drift

**Fix**: Store returned arrays in local variables BEFORE creating Eigen::Maps:

```cpp
auto pose_arr = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
auto coriolis_arr = franka_robot_model_->getCoriolisForceVector();
auto jacobian_arr = franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
Eigen::Map<const Matrix4d> current(pose_arr.data());
Eigen::Map<const Vector7d> coriolis(coriolis_arr.data());
// ...
```

### 2. Rotational Damping Amplifies dq Noise → Reflex

Applying full translational damping (`2√K = 77 Ns/m`) to ALL 6 Cartesian DOF (including rotation) caused `joint_velocity_violation`. The rotational dq noise is amplified through `J^T * D_rot * J_rot * dq_rot` → joint torque noise → vibration → reflex.

**Fix**: Use the multipanda formula for rotational damping: `D_rot = 0.4 × 2√Kr = 0.4 × D_trans`.

### 3. Controller Manager Service Collision

Both `franka_ros2_humble` and `mile_sim` containers run on host networking with default ROS_DOMAIN_ID=0. Each has a `/controller_manager` node. `ros2 control` service calls (like `switch_controllers`) may route to the wrong controller_manager → mode switch fails silently.

**Workaround**: Stop `mile_sim` before mode switching, or shut down its controller_manager.

### 4. Soft-Start Ramp Causes Stiction-Breakaway

The 2-second linear ramp (`ramp = min(1.0, cycle/2000)`) suppresses torque while the filter moves `position_d_` toward the target. When the ramp releases at t=2s, full torque hits → robot breaks stiction violently → velocity spike → reflex.

**Fix**: Remove the ramp entirely (match multipanda behavior). Let the filter naturally ramp torque through error buildup.

### 5. Mode Switch Chicken-and-Egg

`active_mode_ = None` at hardware activation → ALL command interfaces show `[unavailable]`. The controller_manager's pre-check rejects `prepare_command_mode_switch` because interfaces aren't "available." But interfaces can't become available until the mode switch succeeds.

**Fix**: Pre-claim interfaces in `on_activate()`:

```cpp
for (auto& info : command_interfaces_info_) {
    if (info.interface_type == k_HW_IF_CARTESIAN_POSE_COMMAND) {
        info.claim_flag = true;
    }
}
```

This makes CartesianPose interfaces `[available]`, allowing the pose controller to activate and `perform_command_mode_switch` to initialize CartesianPose mode.

### 6. `stopRobot()` Before Torque Init Causes Transient

`perform_command_mode_switch` calls `robot_->stopRobot()` before `initializeTorqueInterface()`. On the FR3, this causes a velocity transient that triggers reflex.

**Fix**: Skip `stopRobot()` for Effort mode (the robot is already stationary).

### 7. Collision Thresholds

The FR3's default collision detection triggers `cartesian_reflex` during impedance motion. Relax thresholds via:

```bash
ros2 service call /service_server/set_full_collision_behavior franka_msgs/srv/SetFullCollisionBehavior \
  "{lower_torque_thresholds_acceleration:[100,...], upper_torque_thresholds_acceleration:[100,...], ...}"
```

## Binary-Search Results: What Causes Reflex?

| Test | Config | Result |
|------|--------|--------|
| Minimal no-op | Zero output, no model calls | **Stable 25+s** ✓ |
| Model reads only | getPoseMatrix/getJacobian/getCoriolis, zero output | **Stable** ✓ |
| Full compute, zero output | All Eigen math, zero torque | **Stable** ✓ |
| K=100, no damping | Simple stiffness, real torque | **Stable** ✓ |
| K=1500, no damping | Higher stiffness, real torque | **Stable** ✓ |
| K=1500 + trans damping | Translation damping only (no rotation) | **Stable** ✓ |
| K=1500 + full 6×6 damping | Same damping on rotation | **REFLEX** ✗ |
| Full multipanda config | K=1500, Kr=100, coriolis ON, Ki=15 | **REFLEX** ✗ |

**Conclusion**: Rotational damping is the reflex trigger. Translation-only impedance (K=1500, no rotation, no coriolis) is stable.

## What Works

1. **Impedance controller HOLDS perfectly** with conservative settings (K=1500, zero rotation, no coriolis)
2. **CartesianPose mode activates cleanly** with pre-claim fix — interfaces show `[available] [claimed]`, zero errors
3. **Pre-claim fix** solves the mode switch chicken-and-egg problem
4. **All controller params are YAML-configurable** — edit `mile_controllers.yaml`, run `colcon build --packages-select mile_franka_controllers`, restart

## What Doesn't Work

1. **Impedance controller can't move the robot useful distances** — any motion >~1cm triggers `joint_velocity_violation` or `power_limit_violation` on this FR3 firmware
2. **CartesianPose controller activates but targets don't move the robot** — the smoothStep trajectory generator doesn't appear to update `hw_cartesian_pose_commands_`
3. **Live mode switching** — FR3 firmware rejects Effort↔CartesianPose switches; must pick one mode at startup

## Key Files Modified

### franka_hardware
- `src/franka_hardware_interface.cpp`: Pre-claim in `on_activate()`, skip `stopRobot()` for Effort, debug prints in `write()`, removed deferred CartesianPose init
- `include/franka_hardware/franka_hardware_interface.hpp`: Removed `deferred_cartesian_pose_init_`

### mile_franka_controllers
- `src/custom_cartesian_impedance_controller.cpp`: Dangling pointer fix, removed soft-start ramp, removed coriolis (optional), deadband (0.5mm), gripper gravity compensation, q/dq from state_interfaces
- `include/mile_franka_controllers/custom_cartesian_impedance_controller.hpp`: `filter_params_=0.008`, `delta_tau_max_=0.5`, configurable damping floors
- `config/mile_controllers.yaml`: All params configurable — pos_stiff, rot_stiff, damping_floor_*, filter_param, delta_tau_max, translational_Ki, translational_clip

### Docker
- `~/franka_ros2/docker-compose.yml`: Added `cpuset: "0-3"` for RT isolation
- `~/mile-franka-tutorial/docker/docker-compose.yml`: Added `cpuset: "4-19"` for mile_sim

## Next Steps

1. **For moving the robot**: Debug why `cartesian_pose_target_controller` targets don't reach the hardware despite interfaces being `[available] [claimed]`. The smoothStep may not be updating `hw_cartesian_pose_commands_` due to interface index mismatch.

2. **For impedance control**: The FR3 firmware's `joint_velocity_violation` thresholds in torque mode appear too tight for useful motion. Options:
   - Check if thresholds can be adjusted in Franka Desk
   - Use incremental sub-millimeter targets (MILE's 10Hz backend already does this)
   - Accept impedance for compliant holding only, use pose control for motion

3. **For eval-real**: Once robot can move, populate `MILE_CONTROLLER=cartesian_pose_target_controller` and run `make eval-real`. The script runs in `mile_sim` — ensure its controller_manager is stopped to avoid service collision.

4. **MoveIt2**: Not installed in current container. Would need Dockerfile update + rebuild. Provides `joint_trajectory_controller` which uses joint position interface (no torque-mode issues).
