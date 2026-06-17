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
1. **Commit the env layer.** `ros_backend.py`, `franka_env.py`, `registration.py`, `mujoco_gt.py`,
   `scripts/train_mile.py` are uncommitted — this is the functional core (incl. the coherence fix).
   Commit them before any further work so the fixes aren't lost.
2. **`make mile` full run is yours to do (by design).** The agent was intentionally asked to wire +
   flow-test only, not run the full pipeline — so the structure is verified but a complete
   N-round sim run hasn't been executed. The `output_dir/franka/policy`/`mental_model` artifacts
   are from the earlier fake-env run, not sim. When you run it, confirm gate 5 (success rate
   improves across rounds) and that no autonomous rollout fires (`auto_eval: false`).
3. **Base-policy "mediocre" decision — RESOLVED.** Two collect verbs now exist:
   `make collect-mediocre` (`--mediocre true --require_success false` → `sim_demos_mediocre.npz`,
   feeds the base policy) and `make collect-expert` (`--mediocre false --require_success true` →
   `sim_demos_expert.npz`, kept as a perfect/reference set). `make base-policy` trains from the
   mediocre set. Per-episode MP4 recording now starts only after `env.reset()` completes and logs
   start/end EE + cube poses for each attempt, fixing stale pre-reset frames at episode starts.
   Still measure the trained base policy's sim success rate to confirm it is mediocre.
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
