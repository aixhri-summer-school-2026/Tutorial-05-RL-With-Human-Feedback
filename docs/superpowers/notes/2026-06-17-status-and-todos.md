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
- **AprilTag-over-ROS pose pipeline (2026-06-18):** RealSense D415 (serial 217222067236)
  color-only 640x480@15Hz, apriltag_ros tag36h11 detection, tf2 chain `panda_link0 →
  tag36h11:0` resolving, `AprilTagPoseSource` returning cube-center poses with <0.2mm
  stability. `RosPoseStampedSource` seam for FoundationPose. Docker passthrough with
  `device_cgroup_rules` for USB + video4linux. 11 unit tests pass (`make pose-test`).
  See `phase2-bringup-checklist.md` §2026-06-18 for the full bring-up report.
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

## 2026-06-18 progress — human-in-the-loop teleop (Phase a, committed)
Teleop data collection with the Xbox gamepad (`intervener: joystick`) is the dev teleop device
alongside the SpaceMouse fallback.
- **`make sim-gui`** (`50b596d`): live MuJoCo window over X11 + NVIDIA GL so the human can see
  when to intervene. **SpaceMouse** `/dev/hidraw` passthrough + `make spacemouse-check` (`34de862`).
  **Gamepad** `JoystickDevice` + `/dev/input` passthrough + `make joystick-check` (`a9ce719`…`13341fc`).
- **Cold-start runbook** (`7368c5b`) and forced classic docker builder for the local hucebot base
  (`a3d666c`) so a fresh machine can rebuild.
- **Collector reworked for a real human** (`collect.py`): stateful **segment toggle**
  (press clutch to start/stop recording an intervention; `intervene` holds across frames so the
  operator can plan), **idle-frame skipping** inside a segment (no do-nothing actions get logged as
  demos), **discard/retry** (Back button throws away a botched episode and re-resets without
  incrementing the count), gripper-state **sync from the policy** on segment entry, and verbose
  per-segment/per-step logging. New `TeleopReading.discard` field; intervener returns a 4-tuple
  `(action, intervene, done, discard)` — `ScriptedIntervener` updated to match.
- **`JoystickDevice` hardened**: segment-toggle mode, gripper toggle (open/close on
  press, `nan` passthrough before first toggle), per-axis sign inversion, and **ghost+bounce button
  filtering** (`hold_confirm_s`, `debounce_s`) to reject phantom spikes/mechanical bounce. Xbox 360
  mapping wired in `train_mile.py`: dx=axis1(-fwd), dy=axis0(-left), dz=axis4(-up), clutch=RB(5),
  gripper=A(0), done=Start(7), discard=Back(6), `translation_scale=0.2`. New `scripts/joystick_identify.py`.
- **`COST_LOOKUP` retuned** for both Franka envs `[250,200]→[70,100]` (lower cost ⇒ model expects the
  human to intervene more readily; seeded guess, still to be calibrated against observed human rate).
- **Grasp robustness** (`ros_backend.py`): grasp force 40→80 N, epsilon 0.02→0.005, and
  `set_gripper` now **waits for both open and close** to physically complete before the EE moves
  (cube was slipping when the arm moved before the gripper shut).
- **Human-paced episodes**: `max_steps` 300→10_000 in the sim env + `max_t=10_000` in the collect
  call, so a human is never truncated mid-stack.
- **`config_franka.json`**: `intervener: joystick`, `num_rounds 2→5`, `num_epochs 5→500`.

### Open before Phase (a) can be called done
- [x] ~~Commit the WIP above~~ — all committed as of 2026-06-18.
- [ ] **Run the full iterative loop** with a human on the gamepad and show success rate improving across
  rounds (finding #2 / gate 5), `auto_eval: false` confirmed (no autonomous rollout).
- [ ] **Measure + record** the mediocre base-policy success rate (`make eval-base`).
- [ ] **Calibrate `COST_LOOKUP`** `[70,100]` against the human intervention rate actually observed.
- [ ] Confirm `num_epochs: 500` per round isn't overfitting the small per-round dataset.

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
**AprilTag pose plan: [2026-06-18-apriltag-ros-pose.md](../plans/2026-06-18-apriltag-ros-pose.md).**
Cost tuning (sim + hardware) is **deferred** per the 2026-06-18 decision — go to the real robot
now, return to [cost tuning](2026-06-18-cost-tuning-guide.md) later; measured improvement is a
later milestone, not a Phase (b) gate.

### Done (2026-06-18 — AprilTag-over-ROS pose pipeline)
- [x] `CameraCalibration` + YAML loader + static-transform-args emission (`pose/calibration.py`)
- [x] `cube_center_pose()` pure half-edge offset helper (`pose/apriltag.py`)
- [x] `AprilTagPoseSource` with injectable tf2 lookup + staleness fallback (`pose/apriltag.py`)
- [x] `RosPoseStampedSource` generic seam — FoundationPose-ready (`pose/ros_posestamped.py`)
- [x] Calibration template (`config/camera_calib.example.yaml`) + apriltag config (`config/apriltag.yaml`)
- [x] Launch file (`launch/apriltag_realsense.launch.py`): realsense2_camera + apriltag_ros + static tf
- [x] Docker apt packages: ros-humble-realsense2-camera, ros-humble-apriltag-ros
- [x] Make verbs: `make pose-test` (11 tests pass), `make apriltag-up`
- [x] **LIVE BRING-UP VERIFIED:** D415 (serial 217222067236) color-only 640x480@15Hz, tag 0 detected
  at decision_margin ~135, tf2 chain `panda_link0 → tag36h11:0` resolves, AprilTagPoseSource
  returns cube-center poses with <0.2mm stability. Docker passthrough (USB + video nodes +
  device_cgroup_rules) works. Findings recorded in `phase2-bringup-checklist.md`.
- [x] tf2 warm-up bug found & fixed: `TransformListener` needs ~500ms priming after construction
  or the first `lookup_transform` misses the latched `/tf_static` message.
- [x] **CONFIRM@bringup offset sign RESOLVED:** the half_edge offset moves INTO the cube (correct
  for this tag mount); no sign flip needed.
- [x] Topics confirmed: `/camera/camera/color/image_raw`, `/detections`, `/tf`, `/tf_static`.

### Remaining for real MILE (in dependency order)

**1. Camera calibration** — `scripts/calibrate_camera.py` (separate follow-on plan).
   Until this exists, `config/camera_calib.yaml` is the identity placeholder and all poses are
   in camera frame, not robot base frame. This is the single hard blocker for base-frame accuracy.
   - Eye-to-hand Charuco-on-gripper: move the arm through a calibration pattern, detect Charuco
     corners, `cv2.calibrateHandEye` → `camera_to_base` in `config/camera_calib.yaml`.
   - Prereq: the hucebot controller must be running on the FR3 so the gripper moves.

**2. Controller bring-up confirmation** — start hucebot's multipanda_ros2 controller docker on the
   FR3 control PC, join the DDS graph from our container (`network_mode: host`), and confirm:
   - [ ] `ros2 node list` shows the controller, franka_gripper_node, move_to_start_example_controller
   - [ ] `/cartesian_impedance/equilibrium_pose` topic exists (our command path)
   - [ ] `/cartesian_impedance/cartesian_pos_curr` topic exists (our EE read path)
   - [ ] `/franka_gripper_node/grasp` action exists (real gripper — currently `GRASP_ACTION` points
     at the sim gripper; the `ros_backend.py` constructor accepts `grasp_action` as a param)
   - [ ] `DOWN_QUAT` gives a down-facing wrist on the real FR3 (the last `CONFIRM@bringup` marker)
   - [ ] `Q_HOME` (move_to_start joint config) clears the workspace

**3. Real env builder** — ✅ DONE (2026-06-18). `_build_real_env()` in `registration.py` wires
   `MultipandaRosBackend(sim=False)` + `AprilTagPoseSource` + `FrankaEnv(mode="real")`.
   Registered as `Franka-Stack-Real-v0`.

**4. Real config** — ✅ DONE (2026-06-18). `config_franka_real.json` with `env_name:
   Franka-Stack-Real-v0`, `intervener: joystick`, `auto_eval: false`, `num_rounds: 5`,
   `episodes_per_round: 3`, `num_epochs: 500`.

**5. COST_LOOKUP** — ✅ DONE (2026-06-18). `Franka-Stack-Real-v0` entry `[70, 100]` added to
   `computational_model.py`.

**6. Real eval** — ✅ DONE (2026-06-18). `scripts/eval_base_policy_real.py` written.
   Operator-gated per-episode confirmations, no autonomous execution, policy runs
   deterministically on `Franka-Stack-Real-v0`. No joystick integration (human supervises
   externally + physical e-stop).

**7. Real MILE training** — once calibration (item 1) and controller bring-up (item 2) exist:
   ```bash
   # On the FR3 control PC: start the hucebot controller docker
   # On our machine:
   make build && make up              # build image, start container
   make shell                          # enter container
   # Inside container:
   make apriltag-up                    # launch camera + AprilTag detection
   # In another shell inside container:
   cd scripts && python3 train_mile.py --config ../config_franka_real.json
   ```
   The human holds the gamepad. Each round: the script collects `episodes_per_round` episodes
   (human teleops each stack, intervening as needed), then trains the policy + mental model
   jointly, then repeats. No autonomous rollout — the policy is never executed without the
   human in the loop during training.

### Eval on real Franka — explicit path (script exists, needs controller + calibration)

```bash
# In-container, with controller + apriltag running:
python3 scripts/eval_base_policy_real.py \
  --policy trained_models/franka/policy \
  --episodes 10
```
The script loads the policy, runs it on `Franka-Stack-Real-v0` for N episodes with
operator-gated per-episode confirmations. The human stands at the robot and watches;
if anything goes wrong they hit the physical e-stop. No autonomous execution —
`auto_eval` is always `false` on hardware. Success = cube stacked without intervention.

Until controller + calibration exist, the closest thing you can do:
- **Verify the pose pipeline:** `make apriltag-up` then `ros2 run tf2_ros tf2_echo panda_link0 tag36h11:0`
- **Verify the gamepad:** `make joystick-check`
- **Verify the controller** (once connected): `ros2 topic echo /cartesian_impedance/cartesian_pos_curr`
- **Sim rehearsal:** `make sim-up && make joystick-check` — practice with gamepad in sim

### Safety invariants (real path — never break these)
- `rollout.auto_eval: false` — the policy is NEVER executed autonomously on hardware
- Operator approval before any arm motion (the script prints what it's about to do and waits)
- Bounded home: `move_to_start_on_reset=True` drives the arm to a known joint config BEFORE
  activating the Cartesian controller (so the equilibrium is captured at a safe posture)
- Gripper completes before EE moves (`set_gripper` waits for physical open/close)
- EE target clamped to `workspace_low`/`workspace_high` (z floor protects the table)
- DDS domain isolation: the real FR3 and our container share `network_mode: host`; verify
  no other ROS nodes are accidentally commanding the same controller

## Phase (c) — France = Vive swap
- `ViveDevice` (thin rclpy subscriber to hucebot `vive_controller` topics) — only France-only code.
- Day-of: launch hucebot controller on their Panda → join DDS → confirm interface names → re-run
  camera calibration → re-tune `COST_LOOKUP` → `intervener: vive`, real env. Rehearse calibration.

## Locked decisions
- Camera: **eye-to-hand** (fixed workspace). Object pose: **AprilTags only** (webcam, CPU; no
  FoundationPose, no cube texturing). Single sim path (fake env dropped as a gate).
