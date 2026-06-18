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

Per the user's call, **intervention-cost tuning is deferred** — we proceed to the real robot now
and return to `COST_LOOKUP` calibration later (sim *and* real). This phase delivers the hardware
path; measured policy improvement is a later milestone once costs are tuned.

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

**In scope:** AprilTag object-pose source behind the existing `ObjectPoseSource` ABC; eye-to-hand
camera calibration script + shared calib file; `Franka-Stack-Real-v0` env builder with real-only
safety semantics; bring-up against the live controller (resolve every `CONFIRM@bringup`, confirm
`DOWN_QUAT`); real gamepad teleop data collection.
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

### 5.2 `AprilTagPoseSource` — object pose
`mile_franka/pose/apriltag.py`. Webcam frame → **pupil-apriltags** detect → tag pose in camera
frame → transform to base frame via `camera→base` extrinsics. One tag id per cube (top/bottom);
apply a fixed tag→cube-center offset. **Pre-grasp only**; expose a `last_seen` staleness signal so
dropouts are detectable.
- **Grasp-transform carry:** once the gripper closes on the top cube the tag is occluded; stop
  trusting it and propagate the held cube's pose from EE motion (grasp offset frozen at grasp time
  — the same "freeze pose at a phase boundary" trick the sim scripted policy already uses).
- Add the AprilTag lib to the image (parent spec §6 reserved this).

### 5.3 Camera calibration — `scripts/calibrate_camera.py`
**Eye-to-hand** (camera fixed, observing the workspace; *not* wrist-mounted). Charuco board
**mounted on the gripper**; drive the arm to N varied poses (teleop or scripted), capture
`(image, O_T_EE)` pairs, detect Charuco, run `cv2.calibrateHandEye` → solve **camera→base**, and
estimate **intrinsics** in the same pass. Output → `config/camera_calib.yaml` (intrinsics +
extrinsics), read by `AprilTagPoseSource`.
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
- ⚠️ **Carry-over risk (OPEN in sim):** wrist down-orientation tracking was unreliable in sim. It
  **must** be resolved on hardware before the scripted grasp is trusted (raise `rot_stiff` /
  `rotational_Ki` and let it settle, or set a known initial joint config, or accept a fixed
  non-vertical approach). This is the single biggest technical risk of the phase. **See §8.2.**
- Re-check grasp force (sim bumped to 80 N) and `action_scale` on the real gripper.

### 5.6 Safety invariants (non-negotiable, every real path)
- `rollout.auto_eval: false` — the policy is **never** executed autonomously on the arm.
- Operator approval before any motion; bounded Cartesian targets; controller clipping on.
- Documented e-stop / "drop everything" procedure in the runbook.

### 5.7 Real data collection
`make collect-mediocre` / `make mile` against `Franka-Stack-Real-v0`, `intervener: joystick`,
`auto_eval: false`. Records the same Box dataset; segment/discard/retry from the phase-(a) collector
rework applies unchanged. **Base policy source is an open decision — §8.3.** Cost tuning is
deferred, so this phase collects data and proves the loop runs; measured improvement comes later.

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
   *(Measured policy improvement is deferred to the cost-tuning milestone, not a gate here.)*

## 8. Open decisions (for the user)

1. **AprilTag detection transport.** In-process **pupil-apriltags** in `AprilTagPoseSource`
   (simplest, CPU, no extra ROS node) **vs** a separate ROS AprilTag node publishing `PoseStamped`
   consumed via `RosPoseStampedSource`. Recommendation: in-process pupil-apriltags now (fewer
   moving parts), keep `RosPoseStampedSource` as the seam so a ROS node / FoundationPose drops in
   later. Confirm?
2. **Wrist-down orientation.** Which resolution path for the OPEN orientation-tracking risk —
   (a) tune `rot_stiff`/`rotational_Ki` + settle time, (b) command a known initial joint config,
   or (c) accept a fixed non-vertical approach and re-tune the scripted grasp? Affects how much
   bring-up time §5.5 needs.
3. **Base policy on hardware.** Transfer the **sim** mediocre base policy to the real robot, or
   **collect fresh real** mediocre demos? Depends on the reality gap; transfer is cheaper, fresh is
   safer. Recommendation: try transfer first, fall back to fresh collection if it behaves wildly.
4. **Camera + mount.** Which webcam, and where is the fixed eye-to-hand mount (must see the whole
   workspace without occluding the arm)? Needed before calibration.
5. **AprilTag family/size + placement** that survives stacking (tag on a cube face that stays
   visible pre-grasp; known tag→center offset). Confirm the printed tag spec.
6. **`camera_calib.yaml` in git?** Commit the format/template but gitignore the machine-specific
   values (they differ per lab and must be recalibrated)? Recommendation: template in git, real
   values gitignored.
