# Phase (b) plan — real FR3 + gamepad + AprilTag (home lab) (2026-06-18)

Design spec (for review): [2026-06-18-phase-b-real-fr3-apriltag-design.md](../specs/2026-06-18-phase-b-real-fr3-apriltag-design.md).

Roadmap: **(a) sim + gamepad → (b) real FR3 + gamepad + AprilTag (lab) → (c) France = Vive swap.**
This plan covers (b): take the validated sim stack to a **real Franka FR3 in the home lab**,
swapping only the object-pose source (MuJoCo GT → AprilTag), adding camera calibration, and a
real env builder. The Cartesian-impedance controller, `FrankaEnv`, teleop (gamepad), collector,
and MILE training are **unchanged** — forward-compat is the standing constraint
([[forward-compat-franka]]): lab → France must be config + re-tune, never a rewrite.

## Preconditions (must be true before starting)
- **A mediocre base-policy artifact exists** (`make collect-mediocre` → `make base-policy`).
  **Cost tuning is deferred** (user's call, 2026-06-18): a measured sim success rate and a
  calibrated sim `COST_LOOKUP` are **not** prerequisites — we go to the real robot now and
  return to tuning later (see [cost tuning guide](../notes/2026-06-18-cost-tuning-guide.md)).
  Measured policy improvement is therefore a later milestone, not a Phase (b) gate.
- WIP committed (collect.py / joystick.py / ros_backend.py etc. — done, `8670bec`).
- hucebot **multipanda_ros2 controller docker** available and runnable against the real FR3
  (the France lab's image — do **not** roll our own controller).
- Hardware on hand: FR3 + gripper, fixed-mount webcam, printed AprilTags (known size), a
  Charuco board for calibration, uniform cubes matching `StackTaskConfig` dims.

## Locked decisions (carried from the design spec)
- Camera: **eye-to-hand** (fixed workspace, camera observes the scene; not on the wrist).
- Object pose: **AprilTags only** (pupil-apriltags, webcam, CPU). No FoundationPose, no cube
  texturing for the tutorial.
- Pose used **pre-grasp**; after grasp, the cube pose is carried by a **grasp-transform**
  (AprilTag may be occluded by the gripper — acceptable, we only need it before grasp).

---

## Workstream 1 — Object pose from AprilTags
Generic ROS pose plumbing first, then the AprilTag detector, behind the existing
`ObjectPoseSource` ABC (`mile_franka/pose/base.py`) so `FrankaEnv` is untouched.

1. **`RosPoseStampedSource(topic)`** (`mile_franka/pose/ros_posestamped.py`): thin lazy-rclpy
   subscriber to a `geometry_msgs/PoseStamped` topic → returns `Pose` in **robot base frame**
   (`panda_link0`). Mirror the lazy-import + `CONFIRM@bringup` pattern from `mujoco_gt.py`.
   This is the seam where any hucebot pose node (AprilTag now, FoundationPose later) plugs in.
2. **`AprilTagPoseSource`** (`mile_franka/pose/apriltag.py`): webcam frame → pupil-apriltags
   detect → tag pose in camera frame → transform to base via `camera→base` extrinsics from
   `config/camera_calib.yaml`. One tag id per cube (top/bottom); apply the known
   tag→cube-center offset. Pre-grasp only; expose `last_seen` staleness so the env/policy can
   detect dropouts.
3. **Grasp-transform carry**: after the gripper closes on the top cube, stop trusting the
   (now-occluded) tag and propagate the cube pose from EE motion (the grasp offset is fixed at
   grasp time — reuse the sim's freeze-bottom-at-place-entry idea from the scripted policy).
4. Add the AprilTag lib to the image (design §6 already reserves this).

## Workstream 2 — Camera calibration (the one France-day asset task; rehearse it)
1. **`scripts/calibrate_camera.py`**: **eye-to-hand** hand-eye calibration. Charuco board
   **mounted on the gripper**; drive the arm to N varied poses (teleop or scripted), capture
   (image, `O_T_EE`) pairs, detect Charuco, run `cv2.calibrateHandEye` → solve **camera→base**.
   Also estimate/record **camera intrinsics** (Charuco) in the same pass.
2. **Output** → `config/camera_calib.yaml` (intrinsics + `camera→base` extrinsics), read by
   `AprilTagPoseSource`. One shared file; document the format inline.
3. **Validation**: place a tag at a tape-measured point in the workspace; confirm
   `AprilTagPoseSource` returns base-frame xyz within ~5–10 mm. Re-measure after any camera
   bump — log a "recalibrate if camera moved" warning.

## Workstream 3 — Real env builder `Franka-Stack-Real-v0`
1. **`_build_real_env()`** in `registration.py` + register `Franka-Stack-Real-v0`:
   `MultipandaRosBackend` against the **real FR3** (not sim) + `AprilTagPoseSource` + `FrankaEnv`.
2. **Real-only env differences** (vs sim builder):
   - `randomize_on_reset=False` — **no** `set_body_state`; the real world isn't ours to teleport.
   - **operator-gated human reset**: `reset()` prompts the operator to physically place the
     cubes and confirm, instead of programmatic randomization.
   - **bounded home** move on reset; conservative workspace bounds (`StackTaskConfig`).
   - `max_steps` human-paced (already 10_000).
3. **Safety invariants on every real path** (non-negotiable):
   - `rollout.auto_eval: false` — the policy is **never** executed autonomously.
   - operator approval before any motion; bounded Cartesian targets; controller clipping on.
   - clear e-stop / "drop everything" story documented in the runbook.

## Workstream 4 — Bring-up against the live controller (resolve every `CONFIRM@bringup`)
Run the **Phase 2 bring-up checklist** (`docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md`)
against the **real** controller, not just sim. Specifically:
1. Start hucebot controller docker on the FR3; join its DDS graph
   (`ros2 node list` sees the controller from our side).
2. Copy the **exact** topic/action/interface names off the running container into
   `ros_backend.py` (equilibrium-pose topic, FrankaState, gripper action, `O_T_EE` topic).
   `grep -rn CONFIRM@bringup mile_franka` — close every marker.
3. **Confirm `DOWN_QUAT` on hardware** (`mile_franka/config.py` — the last open
   `CONFIRM@bringup`): command it as the equilibrium orientation, confirm the gripper points
   at the table. ⚠️ **Carry-over risk:** wrist down-orientation tracking was marked **OPEN in
   sim** (controller wouldn't reliably converge to gripper-down). Resolve or confirm on
   hardware *before* trusting the scripted grasp — see the 2026-06-16 bring-up note.
4. Hand-drive `reset()` → a few `step()`s; confirm obs shape `(18,)`, EE moves, gripper
   opens/closes, and `info['success']` toggles to 1 on a hand-stacked pair.
5. Re-check workspace bounds / `action_scale` / grasp force (the sim bumped grasp to 80 N) on
   the real gripper; tune `StackTaskConfig` if the real cubes differ.

## Workstream 5 — MILE on hardware (sim-to-real transfer first)
1. **Transfer the sim mediocre base policy** to the real robot (load the sim `base_policy` artifact
   directly — identical 18-dim obs space). Watch the first gated rollout; fall back to
   `make collect-mediocre` on hardware only if the reality gap is severe.
2. **Run the full iterative MILE loop on the real robot**: `config_franka.json` pointed at
   `Franka-Stack-Real-v0`, `mode: iterative`, `collector: real`, `intervener: joystick`,
   `auto_eval: false` — collect → retrain → repeat `num_rounds`, using the **seed**
   `COST_LOOKUP['Franka-Stack-Real-v0']` (= sim seed `[70,100]`, separate key).
3. **Cost tuning is deferred** (user, 2026-06-18): run MILE with the seed now; re-calibrate
   `COST_LOOKUP` and prove measured improvement in a later milestone (see cost tuning guide).
4. **Cube geometry:** 5 cm cubes (user decision); set the real env `cube_size=0.05` (tag offset
   0.025 m). For a clean transfer, also set the **sim** env to 0.05 or treat the 1 cm gap as
   reality-gap to watch.

---

## New / changed files (summary)
| File | Change |
|---|---|
| `mile_franka/pose/ros_posestamped.py` | **new** — generic `PoseStamped`→base `Pose` source |
| `mile_franka/pose/apriltag.py` | **new** — `AprilTagPoseSource` (pupil-apriltags + extrinsics) |
| `scripts/calibrate_camera.py` | **new** — eye-to-hand Charuco hand-eye calibration |
| `config/camera_calib.yaml` | **new** — intrinsics + camera→base (gitignore real values?) |
| `mile_franka/envs/registration.py` | add `_build_real_env` + `Franka-Stack-Real-v0` |
| `mile_franka/envs/ros_backend.py` | fill confirmed real interface names; resolve markers |
| `mile_franka/config.py` | confirm/correct `DOWN_QUAT` on hardware |
| `mile/computational_model.py` | add `Franka-Stack-Real-v0` to `COST_LOOKUP` |
| `Makefile` / compose | webcam `/dev/video*` passthrough; real-env make verbs |
| docs | runbook for real bring-up + calibration rehearsal |

## Acceptance gates (Phase b done)
1. `calibrate_camera.py` produces `camera_calib.yaml`; tag at a measured point reads back
   within ~5–10 mm in base frame.
2. `AprilTagPoseSource` tracks both cubes pre-grasp; grasp-transform carries the held cube.
3. `Franka-Stack-Real-v0` `reset()`/`step()` works on the FR3: obs `(18,)`, EE moves, gripper
   actuates, `info['success']` toggles on a real stack.
4. All `CONFIRM@bringup` markers in `mile_franka` resolved; `DOWN_QUAT` confirmed; gripper
   points down reliably enough to grasp.
5. A human can teleop-stack via the gamepad against the real robot; collection records clean
   segments (discard/retry works).
6. A human teleop-stacks via the gamepad on the real robot and the loop records clean segments
   (discard/retry works), **no autonomous rollout** (`auto_eval: false`), safety invariants held
   throughout. *(Measured success-rate improvement is deferred to the cost-tuning milestone, not
   a Phase (b) gate — user's call.)*

## Risks & mitigations
- **Wrist-down orientation unreliable** (carry-over, OPEN in sim) → resolve on hardware first
  (raise `rot_stiff`/`rotational_Ki` and let it settle, or set a known initial joint config);
  blocks grasping if unresolved.
- **AprilTag occlusion at grasp** → only need pose pre-grasp; grasp-transform after (designed in).
- **Camera bump invalidates calibration** → validation check + loud warning; recalibration is
  cheap and rehearsed (it's also the France-day task).
- **Reality gap on the base policy** → be ready to collect a fresh real mediocre policy rather
  than transfer the sim one.
- **Safety** → `auto_eval: false`, operator approval, bounded moves, e-stop documented — every
  real path, no exceptions.

## Explicitly deferred to Phase (c) — France
- `ViveDevice` (thin rclpy subscriber to hucebot `vive_controller` topics) — the only
  France-only code. Everything in this plan must work with the gamepad first.
- Day-of France: launch hucebot controller on their Panda → join DDS → confirm interface
  names → re-run camera calibration → re-tune `COST_LOOKUP` → `intervener: vive`. Rehearse the
  calibration here in the lab so France day is a swap, not a build.
