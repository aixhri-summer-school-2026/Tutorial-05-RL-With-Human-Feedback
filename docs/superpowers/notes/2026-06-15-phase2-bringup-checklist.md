# Phase 2 multipanda bring-up checklist (spec sec 8 gate 1, real backend)

Run when hucebot's multipanda_ros2 controller docker is up (lab MuJoCo sim) and again on
hardware (France). Resolve every `CONFIRM@bringup:` marker:
`grep -rn CONFIRM@bringup mile_franka`.

- [ ] **Controller docker:** start the **hucebot multipanda_ros2 controller image** (the
      France lab's image — do not roll your own). Run our ROS2 client on the same DDS graph
      (shared `ROS_DOMAIN_ID` or `network_mode: host`); confirm `ros2 node list` sees the
      controller node from our side.
- [ ] multipanda sim launched **inside that image**; `ros2 topic list` / `ros2 action list`
      show the equilibrium-pose topic, the FrankaState topic, and the gripper action. Copy
      the exact names off the running hucebot container into the constants in
      `mile_franka/envs/ros_backend.py`.
- [ ] Cube GT poses available: confirm whether the sim publishes per-object PoseStamped
      topics (wire them into `MujocoGtPoseSource`) or whether mjData must be read directly
      (then add that variant). Update `mile_franka/pose/mujoco_gt.py`.
- [ ] Down-facing wrist: publish `DOWN_QUAT` as the equilibrium orientation and confirm
      the gripper points at the table; correct `DOWN_QUAT` in `mile_franka/config.py` if not.
- [ ] Construct `MultipandaRosBackend` + `MujocoGtPoseSource` + `FrankaEnv`; `reset()`
      then `step()` a few hand-chosen actions; confirm obs shape (18,), EE moves, gripper
      opens/closes.
- [ ] Hand-drive a full stack (or replay the `move_to` sequence from
      `scripts/smoke_franka_env.py` against the real backend); confirm `info['success']`
      toggles to 1 when cubes are stacked.
- [ ] Confirm workspace bounds / `action_scale` feel right for the controller (spec sec 10);
      tune `StackTaskConfig` if needed.

## 2026-06-16 calibration findings (live sim, hucebot:franka-humble)

Verified headless end-to-end (build → launch → render → data). Calibration of the
`custom_cartesian_impedance_controller` (`franka_example_controllers`) against the sim:

- **Frames (SOLVED):** sim `get_body_state` poses are WORLD frame; `panda_link0` is at the
  world origin but rotated 180° about z (`quat xyzw=[0,0,1,0]`). So `base=(-wx,-wy,wz)`.
  `mile_franka/envs/ros_backend.py` now transforms reads→base and body-sets→world. EE read
  from body `panda_hand` (panda_hand_tcp / fingers are NOT get_body_state bodies → return 0;
  use TF `panda_link0→panda_hand_tcp` for the true TCP).
- **Position tracking (SOLVED):** with defaults the EE droops/biases (~10 cm y error).
  Setting `pos_stiff=3000`, `translational_clip=0.1`, `ns_stiff_q1_to_4=0`,
  `ns_stiff_q5_to_7=0`, `translational_Ki=30` gives TCP tracking within ~1–2 cm in all axes
  (settle ~4 s). Nullspace stiffness was the main source of the lateral bias.
- **Wrist orientation (OPEN):** getting the gripper to point straight down (approach
  (0,0,-1)) is unreliable. The achieved orientation is gain/config-dependent and slow to
  converge; an orientation sweep under the final gains found nothing closer than ~0.67 of
  unit distance to down (best near `Rx-135`). The earlier "[-0.707,0,0,0.707] → down" result
  held only because nullspace(=2) was pinning the home posture, not because the orientation
  command tracked. NEXT: raise `rot_stiff`/`rotational_Ki` AND hold each pose long enough to
  converge (the 10 Hz moving-target rollout never lets orientation settle), or set a known
  initial joint config, or accept a fixed non-vertical approach. Until this is solved the
  scripted policy aligns over the cube in XY but cannot reliably grasp/stack.
- **Param knobs** live on node `/custom_cartesian_impedance_controller`; settable at runtime
  via `ros2 param set` (reset on relaunch — bake into a launch/param override, not by forking
  franka_bringup).

## 2026-06-16 RESOLVED: scripted expert stacks in sim

Root cause of all the "soft arm / weird pose / no grasp" symptoms: the cartesian impedance
controller activates with its equilibrium target defaulting to the origin (0,0,0) and yanks
the arm into the base, jamming it at a joint limit. Orientation was never the issue
(DOWN_QUAT=[1,0,0,0] is correct; gives approach (0,0,-1) at a sane config).

Fix: the Cartesian controller should be started or cycled **inactive -> active** before a
new rollout. In `on_activate()` the controller captures the current EE pose as its desired
pose; after that the backend commands a bounded move to home. This discards stale targets
and gives one reset/startup path for demo collection and learned-policy inference.
`mile_franka/launch/franka_sim_stacking.launch.py` provides that inactive controller
lifecycle for the sim scene; `MultipandaRosBackend.reset()` owns the activate-at-current
then move-home sequence. On real hardware, use the same contract against the hucebot
controller, but require operator approval before activation/motion. With the sim, the arm
holds the hardware home (joints [0,-0.785,0,-2.356,0,1.571,0.785], EE base
(0.307,0,0.487), gripper straight down), drift ~1mm.

Working recipe (`scripts/franka_sim_grasp_demo.py`): gains pos_stiff=4000,
translational_clip=0.5, translational_Ki=30, rot_stiff=800, ns_stiff_q1_to_4=1.0,
ns_stiff_q5_to_7=0.5; cubes at base x=0.45, y=+-0.12; grasp descend to TCP z~0.035
(cube center), place at z~0.095 (one cube up); Grasp action width=0 force=40 epsilon 0.08.
Result: top cube stacked on bottom (dz=0.06, xy<0.03). Video output_dir/grasp_demo3.mp4.

Backend reset now uses the safe multipanda sequence: deactivate the Cartesian controller
to discard any stale equilibrium target -> unpause -> force gripper open -> run
`move_to_start_example_controller` to the canonical joint ready pose -> randomize cubes ->
activate Cartesian (its `on_activate()` captures the current EE pose) -> command home
through a slow Cartesian interpolation. Do **not** use MuJoCo `/reset` for arm homing in
this ROS 2 stack: `mujoco_ros2_control::Reset()` is a no-op, and `mujoco_ros`'s ROS 2
initial-joint loader is NYI. The launch-time `initial_positions` xacro argument is the
reliable fresh-sim start pose; `move_to_start_example_controller` is the reliable runtime
posture recovery. Real hardware follows the same "discard stale target, hold current pose,
then bounded move" contract, but with operator approval and joint/workspace validation
before any motion. Remaining work: the MILE delta-action policy must descend to
cube-center using the same gains.

## 2026-06-16 — Docker image + sim demo collection bringup

**Image build findings:**
- `USER root` required: base image (`hucebot:franka-humble`) runs as non-root; pip fails without it.
- `xvfb` missing from base image: added `apt-get install -y xvfb` to Dockerfile for headless GLFW rendering.
- Docker `RUN` steps need explicit `source /opt/ros/humble/setup.bash && source /home/user/humble_ws/install/setup.bash` — `bash -lc` does not source `.bashrc`.
- Dep coexistence confirmed: `numpy==2.2.6` works with `gymnasium==0.29.1`; no `numpy<2` pin needed.

**FrankaEnv delta rollout:**
- Rollout script updated to save per-episode video files (`ep0.mp4`, `ep1.mp4`, …) with success filtering and retry logic.
- Default changed to `mediocre=False` (expert demos) with `--require_success true`.
- Scripted policy phase progression confirmed: APPROACH→DESCEND (z≈0.026)→GRASP→LIFT→OVER_BASE→PLACE (z≈0.101, target 0.090)→RELEASE — all 7 phases complete. Success collection expected to work on live run.
- ros_backend.py **was** modified: EE read switched from `get_body_state("panda_hand")` to the controller's `/cartesian_impedance/cartesian_pos_curr` topic (O_T_EE in panda_link0 frame), removing the ~0.103 m `panda_hand` offset that previously drove the arm underground before grasp. The `panda_hand` body is ~0.103 m above the controller EE frame; without this fix the scripted policy commanded the arm into the table.

**MILE iterative run:**
- `config_franka.json` updated to `Franka-Stack-Sim-v0`; all paths confirmed relative to `scripts/`.
- Full `make mile` run deferred to user; pipeline structure verified (no autonomous rollout with `auto_eval: false`).
- Known issue: rclpy "publisher's context is invalid" can occur when `docker exec` reuses a container after a previous rclpy node crashed (SIGKILL leaves stale context). Workaround: restart the `mile_sim` container (`make down && make up`) before running `make mile`.

## 2026-06-18 AprilTag-over-ROS pose pipeline bring-up (D415 + apriltag_ros)

**Hardware:** Intel RealSense D415 (serial 217222067236), USB 2.1 port, third-person
eye-to-hand tripod mount. AprilTags (tag36h11, 36mm) on cube faces. Host: Ubuntu 22.04,
no ROS installed — everything runs in the `mile:franka-humble` docker container with
`network_mode: host`.

### Camera stream

- **Profile:** 640x480@15Hz, RGB8, color-only (depth + IR disabled — the D415 is on a
  USB 2.1 bus, and enabling depth/IR starves the bus causing frame timeouts).
- **Topic:** `/camera/camera/color/image_raw` (realsense2_camera publishes on
  `<camera_name>/color/...` with `camera_name` defaulting to the node name `camera`).
- **Serial:** Must be passed as a STRING (`ParameterValue(serial_no, value_type=str)` in
  the launch file) — realsense2_camera rejects all-digit serials parsed as int.
- **Video minors:** The D415 exposes 6 UVC nodes; minor numbers change after USB reset
  (observed: 3,5,8,9,10,11 → 2,4,5,6,7,9 after driver crash). `device_cgroup_rules:
  'c 81:* rmw'` ensures replug doesn't block container access, but the `devices:` list
  in docker-compose must match the current minors at compose-up time.

### AprilTag detection

- **Tag 0 (bottom cube):** Detected at decision_margin ~135 (very strong). Two
  detections per frame (multi-scale; harmless duplicate).
- **Tag 1 (top cube):** Not detected — cube needs to be positioned with its tag facing
  the camera.
- **Topic:** `/detections` (apriltag_msgs/AprilTagDetectionArray), frame_id
  `camera_color_optical_frame`.
- **Config:** `config/apriltag.yaml` with family=36h11, size=0.036, ids=[0,1],
  frames=["tag36h11:0","tag36h11:1"].

### tf2 transform chain

- **Static transform:** `panda_link0 → camera_color_optical_frame` published from
  `config/camera_calib.yaml` (currently identity placeholder — real calibration pending
  the `calibrate_camera.py` capture script).
- **Dynamic transform:** `camera_color_optical_frame → tag36h11:0` published by
  apriltag_ros on `/tf`.
- **Full chain:** `panda_link0 → tag36h11:0` resolves via tf2_echo and
  `AprilTagPoseSource.get_pose()` after warm-up (see below).
- **tf2 warm-up bug:** `_build_tf2_lookup` originally spun only once for 20ms after
  creating the `TransformListener`, which was insufficient for the listener to receive
  the latched `/tf_static` message. Fixed by spinning 10×50ms before the first lookup
  (commit 1a3910d).

### AprilTagPoseSource validation

- **Stability:** <0.2mm std dev over 5 samples at 1Hz — well within 5-10mm target.
- **Offset direction:** CONFIRMED CORRECT. The half_edge offset (0.025m) is applied
  along the tag's +z axis, which points into the cube (away from the camera). The
  reported cube center moves from tag_z=0.728 to cube_z=0.714 (offset matches the tag's
  tilted +z axis projection). The `CONFIRM@bringup` note in `apriltag.py` about the
  offset sign is RESOLVED — no sign flip needed for this tag mount orientation.
- **Full accuracy check:** Deferred until real camera→base calibration exists. Current
  poses are in camera frame (identity calibration), so only relative/camera-frame
  accuracy is validated.

### Docker configuration notes

- **Video devices are volatile:** The `devices:` list in docker-compose must list the
  current `/dev/video*` minors at compose-up time. A script (`make detect-video`) could
  auto-detect RealSense minors, but for now manually update the list if the container
  fails to start with "no such file or directory."
- **`/dev/bus/usb` mount + `c 189:* rmw`:** Required for libusb device enumeration
  (librealsense uses libusb to set camera parameters before streaming via V4L2).
- **`c 81:* rmw`:** Allows access to video minors that weren't explicitly listed in
  `devices:` (survives replug).
