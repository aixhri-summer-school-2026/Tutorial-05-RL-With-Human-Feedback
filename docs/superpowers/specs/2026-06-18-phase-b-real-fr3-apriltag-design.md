# Phase (b): Real FR3 + Gamepad + AprilTag (home lab)

**Date:** 2026-06-18
**Author:** rayray2002
**Status:** Design (for review)

Second phase of the roadmap **(a) sim + gamepad → (b) real FR3 + gamepad + AprilTag (lab) →
(c) France = Vive**. Parent design: `2026-06-15-mile-franka-stacking-design.md`. Previous phase:
`2026-06-17-phase-a-sim-spacemouse-design.md`. Implementation plan:
`docs/superpowers/plans/2026-06-18-phase-b-real-fr3-apriltag.md`. Status/TODOs:
`docs/superpowers/notes/2026-06-17-status-and-todos.md`. Cost tuning reference:
`docs/superpowers/notes/2026-06-18-cost-tuning-guide.md`.

## 1. Goal

Run the validated MILE-on-Franka stack on a **real FR3 in the home lab**, driven by the **Xbox
gamepad**, with object poses from **AprilTags** instead of MuJoCo ground truth. Prove the loop
that was shown in sim — mediocre base policy, human intervenes via the gamepad, data recorded —
works on hardware end-to-end and safely. This is the dress rehearsal for France: only the teleop
device (gamepad → Vive) and the controller host should change in phase (c).

Per the user's call, **the full MILE iterative loop (collect → retrain → repeat) runs on the real
robot** in this phase — it is *not* a data-collection-only phase. What is **deferred** is
`COST_LOOKUP` *calibration*: MILE runs with the **seed cost** `[70,100]`, and the formal "measured
success-rate improvement across rounds" check is deferred to the later cost-tuning milestone (sim
*and* real). So: MILE trains on hardware now; we tune the costs and prove improvement afterward.

## 2. Why this phase matters (not just plumbing)

The structural risk moves from "does intervention learning work" (a sim question, deferred with
tuning) to **"does the hardware path produce clean data safely"**:
- **Sensing reality gap:** AprilTag pose in the robot base frame is only as good as the camera
  calibration. A few-mm/few-degree extrinsics error turns into failed grasps. This is the first
  time the pipeline depends on calibration rather than perfect GT.
- **Controller reality gap:** the hucebot controller's real interface names, `DOWN_QUAT`, grasp
  force, and (carry-over risk) **reliable wrist-down orientation** must be confirmed on metal.
- **Safety:** the policy must never run autonomously on a real arm; every motion is operator-gated.
These can only be discovered on hardware and are independent of cost tuning, so doing them now is
the right ordering.

## 3. Prerequisites

1. **Phase (a) loop runs in sim** with the gamepad (it does, modulo the full N-round measured
   run). A **mediocre base policy artifact exists** (`make collect-mediocre` → `make base-policy`)
   — MILE needs a policy that starts but fails the precision step. *Not* required: a measured sim
   success rate or sim-tuned `COST_LOOKUP` (deferred).
2. **Phase (a) WIP committed** (done: `8670bec`).
3. **hucebot multipanda_ros2 controller docker** runs against the real FR3 (the France lab's
   image — not a hand-rolled stack).
4. **Hardware on hand:** FR3 + gripper; fixed-mount webcam; printed AprilTags (known family +
   size); a Charuco board for calibration; uniform cubes matching `StackTaskConfig` dims.

## 4. Scope

**In scope:** AprilTag object-pose source (RealSense) behind the existing `ObjectPoseSource` ABC;
eye-to-hand camera calibration script + shared calib file; `Franka-Stack-Real-v0` env builder with
real-only safety semantics; bring-up against the live controller (resolve every `CONFIRM@bringup`,
confirm `DOWN_QUAT`); real gamepad teleop; **the full MILE iterative loop (collect → retrain) on the
real robot** with the seed `COST_LOOKUP`.
**Out of scope:** Vive device + France day-of (phase c); `COST_LOOKUP` tuning (deferred, both sim
and real); FoundationPose / markerless / textured tasks; any change to `FrankaEnv`, the teleop
abstraction, the `Collector`, or the Box dataset schema (all reused unchanged — forward-compat).

## 5. Components

### 5.1 `RosPoseStampedSource(topic)` — generic pose seam
`mile_franka/pose/ros_posestamped.py`. Lazy-rclpy subscriber to a `geometry_msgs/PoseStamped`
topic, returning `Pose` in **robot base frame** (`panda_link0`). Mirrors the lazy-import +
`CONFIRM@bringup` style of `mujoco_gt.py`. This is the abstraction seam: AprilTag now, a hucebot
FoundationPose node later, drop in unchanged (both publish `PoseStamped` — parent spec §5.4).
**Open: in-process detection vs ROS node — see §8.1.**

### 5.2 `AprilTagPoseSource` — object pose (RealSense)
`mile_franka/pose/apriltag.py`. **Intel RealSense** RGB frame (`pyrealsense2`) → **pupil-apriltags**
detect (`tag36h11`) → tag pose in camera frame → transform to base frame via `camera→base`
extrinsics. **Tags already exist:** `scripts/generate_cube_tags.py` prints `tag36h11`, **ID 0 =
bottom cube, ID 1 = top cube**, `tag_size ≈ 0.036 m` (the value the script reports — pass it
verbatim to the detector). Apply a fixed **tag→cube-center offset** (= half the cube edge along the
tagged-face normal, plus any tag-on-face centering offset). **Pre-grasp only**; expose a
`last_seen` staleness signal so dropouts are detectable.
- **Grasp-transform carry:** once the gripper closes on the top cube the tag is occluded; stop
  trusting it and propagate the held cube's pose from EE motion (grasp offset frozen at grasp time
  — the same "freeze pose at a phase boundary" trick the sim scripted policy already uses).
- **Deps/passthrough:** add `pyrealsense2` + pupil-apriltags to the image; map the RealSense **USB**
  device into the container (USB, not `/dev/video*`) — commented compose line like the gamepad one.
- **RealSense bonus:** RGB-D is available, so depth can later refine/validate the tag's z and a
  hucebot **FoundationPose** node could drop into `RosPoseStampedSource` (§5.1) unchanged for a
  future textured task. Not used now — AprilTag-only for the cube tutorial.
- **Cube size — resolved (user, 2026-06-18): 5 cm everywhere.** Matches the tag sheet's 5 cm face,
  so `generate_cube_tags.py` output is used as-is. `StackTaskConfig.cube_size = 0.05` is now the
  default (fake env), the sim builder already used 0.05, and the real env uses it too — done in
  code. Tag→cube-center offset = **0.025 m** half-edge along the tagged-face normal; detector
  `tag_size ≈ 0.036 m`. No sim↔real geometry gap to watch.
- **Tag placement — resolved (user, 2026-06-18): one tag on a single cube face.** Each cube carries
  exactly one `tag36h11` marker on one face (ID 0 bottom, ID 1 top). The pose source reports the
  cube center as `tag_pose ⊕ (0.025 m along the tagged-face inward normal)`. The operator places
  cubes tag-up / tag-toward-camera so the marker is visible to the third-person RealSense pre-grasp;
  occlusion after grasp is handled by the grasp-transform carry.

### 5.3 Camera calibration — `scripts/calibrate_camera.py`
**Eye-to-hand, third-person** (resolved: the RealSense is a **fixed third-person camera** observing
the workspace; *not* wrist-mounted) — placed to see the whole workspace without the arm occluding
the cubes pre-grasp. Because the RealSense
ships **factory intrinsics** (read via `pyrealsense2`), this script is **extrinsics-focused**:
Charuco board **mounted on the gripper**, drive the arm to N varied poses (teleop or scripted),
capture `(image, O_T_EE)` pairs, detect Charuco, run `cv2.calibrateHandEye` → solve **camera→base**.
Read intrinsics from the stream (optionally re-estimate from the Charuco set as a cross-check).
Output → `config/camera_calib.yaml` (intrinsics + extrinsics), read by `AprilTagPoseSource`.
- **Validation:** place a tag at a tape-measured workspace point; confirm base-frame xyz within
  ~5–10 mm. Re-measure / re-run after any camera bump (loud warning).
- This is **the only asset step that cannot be pre-done before France** (parent spec §5.8/§7), so
  it is rehearsed here.

### 5.4 `Franka-Stack-Real-v0` env builder
`_build_real_env()` in `registration.py` + register the id: `MultipandaRosBackend` against the
**real FR3** + `AprilTagPoseSource` + the unchanged `FrankaEnv`. Real-only differences vs the sim
builder:
- `randomize_on_reset=False`; **no** `set_body_state` (the real world isn't ours to teleport).
- **Operator-gated reset:** `reset()` prompts the operator to physically place the cubes and
  confirm, then does a **bounded home** move; conservative `StackTaskConfig` workspace bounds.
- `max_steps` human-paced (already 10_000).

### 5.5 Bring-up against the live controller (resolve every `CONFIRM@bringup`)
Run the Phase 2 bring-up checklist (`notes/2026-06-15-phase2-bringup-checklist.md`) against the
**real** controller: join its DDS graph; copy the **exact** topic/action/interface names off the
running hucebot container into `ros_backend.py` (`grep -rn CONFIRM@bringup mile_franka`); confirm
obs shape `(18,)`, EE motion, gripper actuation, and `info['success']` on a hand-stacked pair.
- **Confirm `DOWN_QUAT`** on hardware (last open marker in `config.py`).
- ✅ **Wrist down-orientation — resolved (user, 2026-06-18): set a known initial joint config (§5.8).**
  Instead of relying on the soft impedance controller to *achieve* a down orientation from an
  arbitrary start, reset the arm to a precomputed wrist-down joint config, then hand over to the
  Cartesian controller (which captures the current — now down — EE on activation). This removes the
  dependence on rotational stiffness/settling that was the root of the sim failure.
- Re-check grasp force (sim bumped to 80 N) and `action_scale` on the real gripper.

### 5.8 Known wrist-down home via joint-space reset (resolves the orientation risk)
Today `_home()` (`ros_backend.py:364`) only ramps a **Cartesian** equilibrium pose
(`position, DOWN_QUAT`) through the impedance controller — the soft path that could not hold the
wrist down in sim. Replace the *start* of homing with a **joint-space** move to a fixed config:
- **`Q_HOME` constant** (7-vector) in `config.py`: a neutral, retracted, **wrist-down** posture
  whose TCP is at the home Cartesian point (~`[0.45, 0.0, table_z+0.33]`) with approach `(0,0,-1)`.
  Compute it **offline via IK** seeded from the Franka "ready" pose
  `[0, -π/4, 0, -3π/4, 0, π/2, π/4]` (already wrist-down) solving for `DOWN_QUAT` at the home point;
  store the resulting joint vector as a constant so sim and real share it (**forward-compat**;
  France inherits the same `Q_HOME`). The ready pose itself is the fallback `Q_HOME` if IK is
  unavailable.
- **Joint-space homing — split by what the multipanda stack actually provides** (verified by
  reading `~/multipanda_ros2`):
  - *real:* `move_to_start_example_controller` **is defined in the real config**
    (`franka_bringup/config/real/single_controllers.yaml`). It drives to a **hardcoded**
    `q_goal = [-0.008, -0.005, 0.011, -1.563, 0.005, 1.603, 0.850]`
    (`move_to_start_example_controller.cpp`; it ignores external goals). So the real builder sets
    `move_to_start_on_reset=True`: reset runs `_move_to_start()` (joint → wrist-down) →
    `_home(activate_controller=True)` (Cartesian captures the down EE, ramps to home with
    `DOWN_QUAT`). `config.Q_HOME` mirrors that `q_goal` for documentation / a future custom move.
  - *sim:* the **sim config has no `move_to_start` controller**
    (`config/sim/single_sim_controllers.yaml`), so this path is unavailable. Sim keeps
    `move_to_start_on_reset=False` and homes via the Cartesian impedance controller alone — which,
    with `SIM_STACKING_GAINS`, already holds the wrist down well enough that scripted/expert
    rollouts stack reliably. (My earlier attempt to re-enable `move_to_start` in sim was reverted:
    it would fail because controller_manager has no type for it in the sim config.)
  - **`CONFIRM@bringup`:** that the real `q_goal` is genuinely wrist-down and clears the workspace
    (refine via IK if not); the exact real gripper namespace (`franka_gripper_node/grasp`).
- **Reset sequence becomes:** open gripper → `move_to_joint_config(Q_HOME)` (wrist now physically
  down) → activate Cartesian impedance (captures the down EE as its equilibrium) → existing
  Cartesian settle. The 10 Hz policy/scripted loop then only ever commands small **position** deltas
  with `DOWN_QUAT` held — starting from a genuinely-down pose, so orientation drift no longer
  prevents grasping.
- Keep the bounded/operator-gated reset semantics (§5.4) and safety invariants (§5.6); a joint-space
  move to a fixed `Q_HOME` is bounded and repeatable by construction.

### 5.6 Safety invariants (non-negotiable, every real path)
- `rollout.auto_eval: false` — the policy is **never** executed autonomously on the arm.
- Operator approval before any motion; bounded Cartesian targets; controller clipping on.
- Documented e-stop / "drop everything" procedure in the runbook.

### 5.7 Real data collection — sim-to-real transfer first
`make mile` against `Franka-Stack-Real-v0`, `intervener: joystick`, `auto_eval: false`. **Decision:
transfer the sim mediocre base policy to the real robot first** (load the sim `base_policy`
artifact directly) rather than collecting fresh real demos — the obs space (`ee_xyz, gripper_width,
top_pose(7), bottom_pose(7)`, 18-dim → 72 after FrameStack) is identical sim↔real by design, so the
policy is loadable as-is. Watch the **first** transfer rollout closely under operator gating; if it
behaves wildly (large reality gap), fall back to `make collect-mediocre` on hardware.

**MILE runs here, on hardware:** after the transferred policy is loaded, the iterative loop
(`mode: iterative`, `collector: real`, `intervener: joystick`, `auto_eval: false`) executes
collect → `InterventionTrainer` retrain → repeat for `num_rounds`, using the **seed**
`COST_LOOKUP[Franka-Stack-Real-v0]` (= the sim seed `[70,100]`). Records the same Box dataset;
segment/discard/retry from the phase-(a) collector applies unchanged. Only the *cost calibration*
and the formal improvement gate are deferred — the training itself happens on the real robot.

## 6. Data flow (unchanged collector, swapped pose source + backend)

Real FR3 (hucebot controller docker) ← `MultipandaRosBackend`. Webcam → `AprilTagPoseSource` →
base-frame cube poses → `FrankaEnv` obs `(18,)`. Base policy drives (ν=0); on gamepad clutch the
human overrides (ν=1); `Collector` records every step; `InterventionTrainer` retrains; repeat.
`auto_eval: false` throughout. Identical to sim except the pose source and robot backend — the
forward-compat payoff.

## 7. Acceptance gates

1. `calibrate_camera.py` produces `camera_calib.yaml`; a tag at a measured point reads back within
   ~5–10 mm in base frame.
2. `AprilTagPoseSource` tracks both cubes pre-grasp; grasp-transform carries the held cube.
3. `Franka-Stack-Real-v0` `reset()`/`step()` works on the FR3: obs `(18,)`, EE moves, gripper
   actuates, `info['success']` toggles on a real stack.
4. All `CONFIRM@bringup` markers resolved; `DOWN_QUAT` confirmed; gripper points down reliably
   enough to grasp.
5. A human teleop-stacks via the gamepad against the real robot; collection records clean segments
   (discard/retry works), `auto_eval: false`, safety invariants held throughout.
6. **The MILE iterative loop runs on the real robot**: transferred base policy → collect → retrain →
   repeat `num_rounds` with the seed `COST_LOOKUP`, no autonomous rollout, completes without manual
   restarts. *(Measured success-rate improvement is deferred to the cost-tuning milestone, not a
   gate here.)*

## 8. Open decisions

1. ~~AprilTag detection transport~~ — **resolved:** in-process **pupil-apriltags** on the RealSense
   RGB stream now (fewest moving parts); `RosPoseStampedSource` (§5.1) kept as the seam so a ROS
   node / FoundationPose drops in later.
2. ~~Wrist-down orientation~~ — **resolved (user, 2026-06-18): known initial joint config** (option
   b), realized via the stack's `move_to_start_example_controller` on **real** (it exists only in
   the real config; `Q_HOME` mirrors its hardcoded `q_goal`). **Sim** keeps Cartesian-only homing
   (no sim move_to_start controller) — already adequate. See §5.8. *Still to confirm at bring-up:*
   that the real `q_goal` is wrist-down and clears the workspace.
3. ~~Base policy on hardware~~ — **resolved (user, 2026-06-18): sim-to-real transfer first.** Load
   the sim mediocre base policy directly (identical obs space); fall back to fresh real collection
   only if the first gated rollout behaves wildly. See §5.7.
4. ~~Camera + mount~~ — **resolved (user, 2026-06-18): Intel RealSense, fixed third-person**
   (`pyrealsense2`, factory intrinsics, RGB-D; eye-to-hand, external, sees the whole workspace).
5. ~~AprilTag family/size / cube edge / placement~~ — **resolved (user, 2026-06-18):** `tag36h11`,
   ID 0=bottom / ID 1=top, `tag_size ≈ 0.036 m` from `generate_cube_tags.py`; **5 cm cubes**
   (`cube_size=0.05`, offset 0.025 m); **one tag on a single face**, placed visible to the
   third-person camera pre-grasp (§5.2).
6. **`camera_calib.yaml` in git?** Commit the format/template but gitignore the machine-specific
   values (they differ per lab and must be recalibrated)? Recommendation: template in git, real
   values gitignored.
