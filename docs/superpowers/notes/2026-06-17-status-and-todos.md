# MILE-on-Franka — status & TODOs (2026-06-17)

Snapshot after the docker increment was implemented by an agent and reviewed. Roadmap phases:
**(a)** sim + gamepad → **(b)** real FR3 + gamepad + AprilTag (lab) → **(c)** France = Vive swap.
(Teleop device is now the **Xbox gamepad / `JoystickDevice`**, not the SpaceMouse; SpaceMouse code is kept as a fallback.)
Forward-compat is a standing constraint (see the `forward-compat-franka` memory): the lab→France
path must be config + re-tune, never a rewrite.

## Done & verified
- Full sim stack: `FrankaEnv`, `MultipandaRosBackend` (sim), `MujocoGtPoseSource`, scripted
  policy, demos→BC base policy, `Collector` + `ScriptedIntervener`/`TeleopIntervener`,
  `SpaceMouseDevice` (code), MILE training wiring, `COST_LOOKUP` entries.
- Sim verified headless end-to-end (build→launch→render→data); waypoint grasp demo stacks.
- Docker increment: image (`mile:franka-humble`), compose, Makefile verbs, `--demos` base policy,
  `config_franka.json`→`Franka-Stack-Sim-v0`. Dep coexistence confirmed (numpy 2.2.6 + gym 0.29.1).
- **Coherence gate solved:** EE read switched to the controller `O_T_EE` topic
  (`/cartesian_impedance/cartesian_pos_curr`), removing the ~0.103 m `panda_hand` offset that
  drove the arm underground before grasp; `franka_env.step` anchors the target to the actual EE.
  Scripted 7-phase progression (APPROACH→…→RELEASE) confirmed in sim.
- Sim controller gains tuned down for less shaking while preserving 10 Hz policy semantics:
  `pos_stiff=2500`, `translational_clip=0.2`, `translational_Ki=0`, `rot_stiff=600`,
  `rotational_Ki=0`, nullspace stiffness `0/0`. Live expert rollout still succeeds.
- Placement weaving/hard contact root cause: scripted policy was commanding the EE to the
  desired top-cube center. It now freezes the bottom pose at place entry and commands
  `EE = desired_top_center + grasp_offset`; validation reduced bottom-cube xy displacement
  from ~15 mm to ~3 mm on the checked rollout.
- Sim reset/workspace moved farther forward: sim cube centers now sample x≈0.45–0.70 and
  y≈±0.13 after reset margin. Scripted xyz action slew limited to `0.6` per 10 Hz tick for
  smoother phase transitions; forward/centered expert rollout succeeded.
- Full-trajectory check: raised scripted transit hover from 0.20 m to 0.30 m so approach and
  over-base are mostly horizontal, with vertical motion isolated to descend/place. Phase
  metrics improved (approach excess path 2.58→1.95, place 1.31→1.15) while preserving success.
- Densified scripted waypoints: each 10 Hz action now targets an intermediate Cartesian
  waypoint capped at 3.5 cm. Forward rollout succeeded; total path excess improved 4.78→4.38,
  action saturation dropped to zero, and bottom-cube displacement dropped 8.4→5.6 mm, at the
  cost of a longer rollout (~41 s MP4). Descend still has the worst direction reversal.
- Transition pause tune: relaxed scripted `pos_tol` from 12 mm to 18 mm after the densify
  change. This removed near-waypoint static waiting (256→172 steps, MP4 back to ~28 s) while
  keeping zero action saturation and similar path metrics.
- Table/overshoot tune: descend target now floors EE z at 4.5 cm to avoid table contact, and
  waypoint speed tapers only near DESCEND/PLACE goals (`slow_radius=9 cm`, fraction 0.45).
  Live rollout succeeds with min EE z≈4 cm and bottom-cube displacement ~4.6 mm, trading speed
  for contact gentleness (~36.5 s MP4, 228 steps).
- Phase-specific tolerances: loose free-space APPROACH/LIFT tolerance (4.5 cm), precise GRASP
  descent (1.2 cm), precise pre-place centering (1.6 cm), and precise top-cube PLACE tolerance
  (1.0 cm). Validation: success in 214 steps, min EE z≈4.2 cm, bottom-cube displacement ~0.2 mm.
- Phase-specific tolerances: loose free-space transit/lift tolerance 3.5 cm, tight grasp/place
  contact tolerance 1 cm. This preserves precision where contact matters while avoiding
  free-space over-waiting; validation improved 228→207 steps, bottom displacement 4.6→0.8 mm,
  and MP4 duration to ~33.3 s.

## Review findings to resolve (carry-over from the docker increment)
1. **Commit the env layer — DONE** (`abb606b`). `ros_backend.py`, `franka_env.py`,
   `registration.py`, `mujoco_gt.py` + assets/config committed, coherence `O_T_EE` fix included.
2. **`make mile` full run is yours to do (by design).** Still the open gate — see the
   2026-06-18 section: the iterative human-in-the-loop loop now *runs* with the joystick, but a
   clean N-round run that demonstrates gate 5 (success rate improves across rounds) with
   `auto_eval: false` has not been signed off yet.
3. **Base-policy "mediocre" decision — RESOLVED.** `make collect-mediocre`/`make collect-expert`
   verbs exist; `make base-policy` trains from the mediocre set; `make eval-base` (`8d2164d`)
   measures the trained base policy's sim success rate. Still: record the actual measured number.
4. **Fix the contradictory note — DONE** (`e266242`). Bring-up note corrected to say the
   `O_T_EE` coherence fix lives in `ros_backend`/`franka_env`.
5. **rclpy context fragility — DONE** (`88bf7cf`). Idempotent rclpy init/shutdown; `make mile`
   reruns no longer need `make down && make up`.

## 2026-06-18 progress — human-in-the-loop teleop (Phase a, in flight; uncommitted WIP)
Focus shifted to making real human teleop data collection actually usable, using an Xbox
gamepad (`intervener: joystick`) as the dev teleop device alongside the SpaceMouse path.
- **`make sim-gui`** (`50b596d`): live MuJoCo window over X11 + NVIDIA GL so the human can see
  when to intervene. **SpaceMouse** `/dev/hidraw` passthrough + `make spacemouse-check` (`34de862`).
  **Gamepad** `JoystickDevice` + `/dev/input` passthrough + `make joystick-check` (`a9ce719`…`13341fc`).
- **Cold-start runbook** (`7368c5b`) and forced classic docker builder for the local hucebot base
  (`a3d666c`) so a fresh machine can rebuild.
- **Collector reworked for a real human** (`collect.py`, uncommitted): stateful **segment toggle**
  (press clutch to start/stop recording an intervention; `intervene` holds across frames so the
  operator can plan), **idle-frame skipping** inside a segment (no do-nothing actions get logged as
  demos), **discard/retry** (Back button throws away a botched episode and re-resets without
  incrementing the count), gripper-state **sync from the policy** on segment entry, and verbose
  per-segment/per-step logging. New `TeleopReading.discard` field; intervener returns a 4-tuple
  `(action, intervene, done, discard)` — `ScriptedIntervener` updated to match.
- **`JoystickDevice` hardened** (uncommitted): segment-toggle mode, gripper toggle (open/close on
  press, `nan` passthrough before first toggle), per-axis sign inversion, and **ghost+bounce button
  filtering** (`hold_confirm_s`, `debounce_s`) to reject phantom spikes/mechanical bounce. Xbox 360
  mapping wired in `train_mile.py`: dx=axis1(-fwd), dy=axis0(-left), dz=axis4(-up), clutch=RB(5),
  gripper=A(0), done=Start(7), discard=Back(6), `translation_scale=0.2`. New `scripts/joystick_identify.py`.
- **`COST_LOOKUP` retuned** for both Franka envs `[250,200]→[70,100]` (lower cost ⇒ model expects the
  human to intervene more readily; seeded guess, still to be calibrated against observed human rate).
- **Grasp robustness** (`ros_backend.py`, uncommitted): grasp force 40→80 N, epsilon 0.02→0.005, and
  `set_gripper` now **waits for both open and close** to physically complete before the EE moves
  (cube was slipping when the arm moved before the gripper shut).
- **Human-paced episodes**: `max_steps` 300→10_000 in the sim env + `max_t=10_000` in the collect
  call, so a human is never truncated mid-stack.
- **`config_franka.json`**: `intervener: joystick`, `num_rounds 2→5`, `num_epochs 5→500`.

### Open before Phase (a) can be called done
- **Commit the WIP above** (collect.py, joystick.py, base.py, ros_backend.py, registration.py,
  train_mile.py, config_franka.json, computational_model.py, Makefile, compose, joystick_identify.py).
- **Run the full iterative loop** with a human on the gamepad and show success rate improving across
  rounds (finding #2 / gate 5), `auto_eval: false` confirmed (no autonomous rollout).
- **Measure + record** the mediocre base-policy success rate (`make eval-base`).
- **Calibrate `COST_LOOKUP`** `[70,100]` against the human intervention rate actually observed.
- Confirm `num_epochs: 500` per round isn't overfitting the small per-round dataset.

## Phase (a) — sim + SpaceMouse/gamepad  (in flight; see 2026-06-17-phase-a-sim-spacemouse-design.md)
- [x] Live MuJoCo rendering (X11 passthrough, `make sim-gui`) so a human can see when to intervene.
- [x] SpaceMouse passthrough (`/dev/hidraw*`) + `make spacemouse-check`. Gamepad `/dev/input`
      passthrough + `make joystick-check` added as the dev teleop device.
- [x] `intervener: spacemouse|joystick` wired; `JoystickDevice` deadband→ν, segment toggle,
      gripper toggle, button filtering verified offline.
- [~] Genuinely-mediocre base policy (finding #3): verbs + `make eval-base` exist; success number
      still to be recorded.
- [ ] Calibrate `COST_LOOKUP` `[70,100]` for `Franka-Stack-Sim-v0` against observed human rate.
- [ ] Run N=5 rounds with a human on the gamepad; show success rate improves across rounds.

## Phase (b) — real FR3 + gamepad + AprilTag (lab)
**Design spec (for review): [2026-06-18-phase-b-real-fr3-apriltag-design.md](../specs/2026-06-18-phase-b-real-fr3-apriltag-design.md).**
**Plan: [2026-06-18-phase-b-real-fr3-apriltag.md](../plans/2026-06-18-phase-b-real-fr3-apriltag.md).**
Cost tuning (sim + hardware) is **deferred** per the 2026-06-18 decision — go to the real robot
now, return to [cost tuning](2026-06-18-cost-tuning-guide.md) later; measured improvement is a
later milestone, not a Phase (b) gate. Open decisions for review live in §8 of the spec.
- Generic `RosPoseStampedSource(topic)` → `AprilTagPoseSource` (pupil-apriltags, webcam),
  returning `Pose` in robot base frame.
- `scripts/calibrate_camera.py` — **eye-to-hand**, Charuco-on-gripper, `cv2.calibrateHandEye` →
  shared `config/camera_calib.yaml` (intrinsics + camera→base).
- Real env builder `Franka-Stack-Real-v0`: `MultipandaRosBackend` against the FR3,
  `randomize_on_reset=False`, operator-gated human reset, bounded home, no sim `set_body_state`.
- Confirm `DOWN_QUAT` on hardware (last `CONFIRM@bringup`); copy any differing controller
  interface names into `ros_backend.py`.
- Safety invariants on every real path: `rollout.auto_eval: false`, operator approval, bounded move.

## Phase (c) — France = Vive swap
- `ViveDevice` (thin rclpy subscriber to hucebot `vive_controller` topics) — only France-only code.
- Day-of: launch hucebot controller on their Panda → join DDS → confirm interface names → re-run
  camera calibration → re-tune `COST_LOOKUP` → `intervener: vive`, real env. Rehearse calibration.

## Locked decisions
- Camera: **eye-to-hand** (fixed workspace). Object pose: **AprilTags only** (webcam, CPU; no
  FoundationPose, no cube texturing). Single sim path (fake env dropped as a gate).
