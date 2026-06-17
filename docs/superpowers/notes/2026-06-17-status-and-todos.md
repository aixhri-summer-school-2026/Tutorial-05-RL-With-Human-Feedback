# MILE-on-Franka — status & TODOs (2026-06-17)

Snapshot after the docker increment was implemented by an agent and reviewed. Roadmap phases:
**(a)** sim + SpaceMouse → **(b)** real FR3 + SpaceMouse + AprilTag (lab) → **(c)** France = Vive swap.
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

## Review findings to resolve (carry-over from the docker increment)
1. **Commit the env layer.** `ros_backend.py`, `franka_env.py`, `registration.py`, `mujoco_gt.py`,
   `scripts/train_mile.py` are uncommitted — this is the functional core (incl. the coherence fix).
   Commit them before any further work so the fixes aren't lost.
2. **`make mile` full run is yours to do (by design).** The agent was intentionally asked to wire +
   flow-test only, not run the full pipeline — so the structure is verified but a complete
   N-round sim run hasn't been executed. The `output_dir/franka/policy`/`mental_model` artifacts
   are from the earlier fake-env run, not sim. When you run it, confirm gate 5 (success rate
   improves across rounds) and that no autonomous rollout fires (`auto_eval: false`).
3. **Base-policy "mediocre" decision.** `make collect` uses `--mediocre false --require_success true`
   → a *competent* base policy, contradicting spec §5.5 (deliberately mediocre). Measure the BC
   policy's sim success rate; if it is too high, deliberately degrade it (fixed offset / release-high
   / action noise) so interventions matter. Decide and document.
4. **Fix the contradictory note.** The 2026-06-16 bring-up note says "ros_backend.py not modified /
   no offset fix needed" — false; the code has the `O_T_EE` offset fix. Correct it.
5. **rclpy context fragility.** Repeated `make mile` needs `make down && make up` (stale context after
   a killed node). Add proper rclpy init/shutdown lifecycle so reruns don't require a restart.

## Phase (a) — sim + SpaceMouse  (next; see 2026-06-17-phase-a-sim-spacemouse-design.md)
- Live MuJoCo rendering (X11 passthrough, `make sim-gui`) so a human can see when to intervene.
- SpaceMouse passthrough (`/dev/hidraw*`) + in-container device-read sanity check.
- `intervener: spacemouse`; verify `SpaceMouseDevice` deadband→ν, translation_scale, gripper, done.
- Genuinely-mediocre base policy (resolve finding #3).
- Calibrate `COST_LOOKUP` `[cost, cdf_scale]` for `Franka-Stack-Sim-v0` against observed human
  intervention rate.
- Run N=2–3 rounds; show success rate improves across rounds.

## Phase (b) — real FR3 + SpaceMouse + AprilTag (lab)
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
