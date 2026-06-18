# MILE on Franka — Block-Stacking Tutorial (Sim-in-Lab → Hardware-in-France)

**Date:** 2026-06-15
**Author:** rayray2002
**Status:** Design (approved for spec write-up)

## 1. Context & Goal

Use MILE (Model-based Intervention Learning, https://liralab.usc.edu/mile/) as the
hands-on tutorial for the AI–Human–Robot summer school
(https://aihumanrobot.sciencesconf.org). The target hardware is a **Franka Panda**
controlled via the hucebot **multipanda_ros2** stack
(https://github.com/hucebot/multipanda_ros2). We will run MILE's iterative
intervention loop for **N = 2–3 rounds** on a **block-stacking** task.

The host lab (where prep happens) has a **SpaceMouse**; the summer-school lab has an
**HTC Vive** already wired into their stack. The central strategy:

> multipanda_ros2 runs the **same Cartesian-impedance controller in MuJoCo sim and on
> real hardware**. So we build and fully validate the entire pipeline against the
> **simulated Panda + SpaceMouse in the host lab**. In France the only changes are:
> launch against real hardware, swap the teleop device to Vive, add real object-pose
> sensing (AprilTag), and re-tune the intervention cost. Nothing structural changes.

Everything must be **packable into a Docker image** so the prepared environment ships
to France reproducibly.

## 2. Key facts about the MILE codebase (verified by reading the code)

These constrain the design and were confirmed against the current repo:

- **The repo has no real human-in-the-loop teleop.** `iterative_training` →
  `collect_synthetic_data` simulates the human with three pretrained models: a
  `rollout_policy`, an expert `intervention_policy`, and a `gt_mental_model` +
  `computational_intervention_model`. All in MetaWorld sim.
- **On hardware the human replaces both the `intervention_policy` and the
  `gt_mental_model`.** Neither is needed. The mental model is still *trained* (it is
  half the loss) but is not supplied as ground truth.
- **What the trainer actually consumes** (continuous/`Box` path,
  `InterventionTrainer._train_one_epoch`, `mile/algorithm.py`): per batch it uses only
  `state`, `action`, `intervention`. The dataset's `intervention_prob` is read then
  **immediately overwritten** by `computational_intervention_model`
  (`mile/algorithm.py:205`), so the real collector does not need a meaningful one.
- **Loss** (`mile_cont_loss_fn`, `mile/algorithm.py:116-145`): BCE on the intervention
  flag ν + Gaussian NLL of the human action, the latter **only on timesteps where ν=1**.
- **Required dataset dict (Box)** — same shape as `collect_synthetic_data` produces:
  `state, rollout_action, action, intervention_prob, intervention, reward, next_state,
  done`. `prepare_dataset` (`mile/utils.py:44`) stacks **all** keys with `np.array`, so
  every key must be array-convertible with consistent shape.
- **The trainer auto-rolls-out the policy on the env.** `InterventionTrainer.__init__`
  **always** calls `generate_rollout(self.policy, self.env, ...)` (`mile/algorithm.py:186`,
  default 10 episodes), and `rollout.enabled`/`validate.enabled` trigger more during
  training. On hardware this autonomously executes the policy — undesirable/unsafe and
  slow. **Must be guarded behind a config flag for real-robot runs.**
- **`env.step` must return `info['success']`** — `generate_rollout` reads it
  (`mile/algorithm.py:97`).
- **`COST_LOOKUP`** (`mile/computational_model.py:14`) must contain an entry
  `[cost, cdf_scale]` for the env name; the trainer raises otherwise. We add one for the
  Franka task, starting from `pick-place-v2 = [250, 200.0]`.
- **Frame stacking**: `train_mile.py` applies `FrameStack(4)+FlattenObservation` only on
  the MetaWorld branch. Our custom env must get the same wrappers (paper concatenates 3
  previous + current frame).
- **Policy / mental model types**: use `bc` (`ActorCriticPolicy`, net `[256,256]`,
  `NormalizeFeaturesExtractor`+`RunningNorm`). The `rollout_policy` must be a
  differentiable Gaussian policy — a scripted controller cannot be used directly; it must
  be BC-distilled.

## 3. Scope

**In scope**
- A real/sim `FrankaEnv` (gym) over multipanda_ros2.
- Device-abstracted teleop (SpaceMouse now, Vive in France).
- Object-pose abstraction (MuJoCo GT in sim, AprilTag on real).
- Scripted stacking policy → BC base ("mediocre") policy.
- A unified human-in-the-loop collector (demo + intervention modes) replacing the
  synthetic collector.
- Minimal, surgical edits to `train_mile.py` / `algorithm.py` to support the real path.
- Block-stacking task assets for MuJoCo.
- A Docker image packaging the whole stack.

**Out of scope (optional stretch)**
- Image-based observations / CNN policies (we use state-based).
- FoundationPose (AprilTag is primary; FoundationPose is an optional markerless upgrade).
- Peg-insertion (kept as a later second task; block-stacking is first).
- Retiring MetaWorld sim support (left intact; the Franka path simply must not require it).

## 4. Task: Block Stacking

Stack the top cube onto the bottom cube.

- **Action space**: 4-DoF `[dx, dy, dz, gripper]`, end-effector points down (identical to
  MILE/MetaWorld today). Deltas integrated into a Cartesian target.
- **Observation**: `EE pose + gripper width + top-cube pose + bottom-cube pose`
  (MuJoCo ground-truth in sim; AprilTag pre-grasp + grasp-transform after grasp on real;
  bottom cube fixtured or tagged). Then `FrameStack(4)+Flatten`.
- **Success** (`info['success']`): top cube resting on bottom cube, horizontal center
  offset < threshold, vertical position ≈ one cube height, gripper released.
- **Reset**: sim randomizes cube poses within a workspace box; real is human-gated
  ("place two cubes, press Enter").
- **`COST_LOOKUP`**: start `[250, 200.0]`, tune.
- **Why it's a good first task**: real assets are just uniform cubes (buy, no 3D
  printing); alignment + release-height precision makes interventions clearly matter;
  easy, fast resets.

## 5. Architecture

### 5.1 Components

```
                      ┌─────────────────────────────────────────────┐
                      │            mile (training side)              │
  scripted_stack ──►  demos ──► BC ──► base ActorCriticPolicy        │
                      │                     │                        │
   TeleopDevice ──►  Collector(mode) ──► MILE dataset ──► Trainer ──►│ policy + mental model
  (SpaceMouse/Vive)   │      ▲                                       │
                      │      │ obs / step / reset / info['success']  │
                      │  FrankaEnv (gym, rclpy client)               │
                      │      │  ObjectPoseSource (MujocoGT/AprilTag)  │
                      └──────┼───────────────────────────────────────┘
                             │ ROS2 DDS (topics/actions)
                      ┌──────┴───────────────────────────────────────┐
                      │  multipanda_ros2  (MuJoCo sim OR real Panda)  │
                      │  /cartesian_impedance/equilibrium_pose,       │
                      │  FrankaState broadcaster, gripper action srv  │
                      └──────────────────────────────────────────────┘
```

### 5.2 `FrankaEnv` (gym.Env, rclpy node)
- `step(action)`: integrate `[dx,dy,dz]` delta into a target pose, publish to
  `/cartesian_impedance/equilibrium_pose` (`geometry_msgs/PoseStamped`); drive the
  gripper via its action server from the gripper command; wait one control period
  (~10 Hz); read new EE pose from the FrankaState broadcaster; build obs; compute
  reward + `info['success']`; return.
- `reset()`: first put the robot into a safe, consistent start state shared by demo
  collection and learned-policy inference. Sim → launch with a known xacro
  `initial_positions` ready pose; for each rollout, unpause MuJoCo, force the gripper
  open, run multipanda's `move_to_start_example_controller` for runtime joint-posture
  recovery, randomize cube poses, activate the Cartesian controller so its `on_activate()`
  captures the current EE pose, then command a bounded Cartesian move to the policy home.
  Do **not** use MuJoCo `/reset` for arm homing in the ROS 2 stack: `mujoco_ros2_control`
  reset does not restore the Panda posture, and the generic ROS 2 initial-joint loader is
  NYI. Real → human-gated object reset and operator approval before enabling autonomous
  policy execution; same invariant that stale Cartesian targets are discarded and motion to
  home is bounded before the policy can command deltas.
- Registered via `gym.register` so `FrameStack(4)+FlattenObservation` apply.
- `mode='sim'|'real'` selects only the `ObjectPoseSource` and reset behavior; the control
  interface is identical (this is the whole point).

### 5.3 `TeleopDevice` (abstract)
- Interface: `read() -> (action[dx,dy,dz,gripper], intervene: bool, done: bool)`.
- `SpaceMouseDevice` (pyspacemouse / hidapi): `intervene=True` when the puck exceeds a
  deadband or a hold-button is pressed. Mirror the Vive driver's clutch + delta-translation
  semantics so behavior is identical across labs.
- `ViveDevice` (France): subscribe to **hucebot/vive_controller**
  (https://github.com/hucebot/vive_controller). It already publishes
  `/vive/right/pose` (`geometry_msgs/PoseStamped`) and `/vive/right/joint_states`
  (trigger 0-1, grip, trackpad, menu), ships a **clutch-based Teleop Bridge** (delta
  translation + mirrored rotation, axis alignment, linear scaling), has **explicit
  Franka support** (`make franka`), and is Dockerized. `ViveDevice` is therefore a thin
  subscriber, and the **clutch/trigger maps directly to MILE's intervention flag ν**.
  Same `TeleopDevice` interface → zero collector changes. This de-risks the France side:
  hucebot already maintains the teleop driver.

### 5.4 `ObjectPoseSource` (abstract)
- `MujocoGtPoseSource` (sim): read body poses from the sim state.
- `AprilTagPoseSource` (real): `apriltag_ros` / pupil-apriltags, camera intrinsics +
  hand-eye extrinsics; pose used pre-grasp, then grasp-transform after grasp.
  **Primary choice for stacking**: cubes are textureless + symmetric (a near-worst case
  for appearance-based pose tracking), so a fiducial is more robust and far simpler.
- `FoundationPosePoseSource` (optional, markerless upgrade for a *later, textured*
  task): use **hucebot/FoundationPoseNode**
  (https://github.com/hucebot/FoundationPoseNode), hucebot's own ROS2 wrapper
  (YOLO/SAM3 + FoundationPose) that publishes 6-DoF `PoseStamped` — same message type
  as the AprilTag source, so it drops into this abstraction unchanged. Needs GPU+CUDA,
  RGB-D (RealSense), and a `.obj` mesh. Low-friction in France because hucebot maintains
  it, but **not used for the cube-stacking tutorial** (overkill and ill-suited to plain
  cubes).
- Both real sources publish `PoseStamped`, confirming the abstraction: in France we point
  `ObjectPoseSource` at whichever hucebot node is running.
- Object nominal dimensions live in **one shared config** so sim MJCF and real cubes stay
  consistent.

### 5.5 Scripted policy + BC base policy
- `scripted_stack_policy` (sim, uses GT poses): state machine
  *above-pick → descend → close → lift → above-base → descend → open*.
- The scripted policy must be parameterized by the active `FrankaEnv`'s
  `StackTaskConfig`. The fake backend uses the tutorial defaults (`cube_size=0.04`,
  `table_z=0.02`); the multipanda MuJoCo stacking scene uses `cube_size=0.06`,
  `table_z=0.0`. Mixing these values causes wrong grasp/release heights even when the
  controller is healthy.
- Deliberately **mediocre**: fixed wrong offset / release too high / coarse alignment /
  action noise so it starts but cannot reliably finish (the regime MILE needs).
- Generate `(obs, action)` demos → **BC-train** an `ActorCriticPolicy` (imitation BC).
  The scripted controller is *not* used as the rollout policy directly (MILE needs a
  differentiable Gaussian).

### 5.6 `Collector` (unified, replaces `collect_synthetic_data`)
- `mode='demo'`: teleop always in control; record `(obs, action)` only (BC data). Used
  only if we ever want human demos; default demo source is the scripted policy.
- `mode='intervention'`: base policy in control by default; on `intervene=True`, executed
  action = teleop action, ν=1; else ν=0. Record the **full MILE dict** every timestep
  (interventions and non-interventions). `intervention_prob` filled with placeholder
  `[1-ν, ν]`. `k=3` trajectories/round.
- `scripted_intervener` (optional, testing): a scripted "human" that grabs control when
  the base policy drifts past a threshold — lets us validate the collector headless,
  with no person/device attached.

### 5.7 `train_mile.py` / `algorithm.py` edits (surgical)
- In `iterative_training`: swap `collect_synthetic_data` → `Collector(mode='intervention')`.
- Apply `FrameStack(4)+FlattenObservation` to the custom env (generalize the wrapper
  application beyond the MetaWorld branch).
- Add the env's `COST_LOOKUP` entry.
- **Guard the autonomous `generate_rollout`**: gate the `__init__` rollout and the
  training-loop rollouts behind a config flag (e.g. `experiment.rollout.auto_eval`),
  defaulting off for real-robot. Replace with optional human-judged success logging.
- Keep MetaWorld path working; ensure the Franka path does not import/require MetaWorld.

### 5.8 Assets
- `assets/stacking.xml` (or include into multipanda's world XML): two cube bodies with
  free-joints, tuned mass/friction; cube width < gripper max open. Reuse robosuite /
  MuJoCo Menagerie cube props.
- Shared object-dimension config drives both MJCF and real cube selection.
- Sim-validation checklist (lab acceptance gate before policy work):
  1. scene loads, cubes rest stably;
  2. gripper can grasp (width + friction);
  3. object pose readable from sim;
  4. success region defined + verified;
  5. reset randomization range verified.
- Real assets for France: uniform cubes; printed AprilTags (known size, mounted clear of
  grasp); webcam (or RealSense if FoundationPose kept open) + mount + checkerboard/Charuco
  for calibration. **Camera calibration is the only asset step that cannot be pre-done in
  the lab** — script and rehearse it.

## 6. Docker packaging

Goal: one reproducible image containing the full stack so it ships to France.

- **Base**: the hucebot multipanda_ros2 image (ROS 2 Humble, Ubuntu 22.04, Python 3.10 —
  matches MILE's Python 3.10 — and the MuJoCo sim). `rclpy` comes from this base.
- **Added layer** (`docker/Dockerfile`, `FROM` the multipanda image):
  - pip install the MILE stack pinned: `gymnasium==0.29.1`, `stable-baselines3<=2.3.2`,
    `imitation`, `torch`, `wandb`, plus `pyspacemouse`/`hidapi` and an AprilTag lib.
  - `pip install -e .` for the `mile` package.
  - Avoid pulling MetaWorld (mujoco<3) on the Franka path; keep it optional/extra so the
    pinned sim stack doesn't conflict with the ROS/MuJoCo sim.
- **Runtime** (`docker/docker-compose.yml`) for passthrough:
  - SpaceMouse: `/dev/hidraw*` device passthrough (or USB).
  - GUI (MuJoCo viewer): X11 (`DISPLAY`, `/tmp/.X11-unix`).
  - ROS 2 DDS: `network_mode: host` or a shared `ROS_DOMAIN_ID`.
  - GPU (torch training): nvidia runtime (`gpus: all`) — optional, the MLP trains on CPU.
  - Camera (France): `/dev/video*` / RealSense passthrough.
  - Volume for saved models + datasets.
- Produce a locked requirements file for reproducibility.
- **Decision**: single extended image (satisfies "one Docker image") layered on the
  multipanda base, orchestrated by compose for device/GUI/GPU passthrough. Alternative
  (two containers communicating over DDS) is noted but not chosen — single image is
  simpler to ship for a tutorial.

## 7. Lab → France split

| Concern | Host lab (prep) | France (hardware) |
|---|---|---|
| Robot backend | multipanda MuJoCo sim | real Panda |
| Teleop | `SpaceMouseDevice` | `ViveDevice` |
| Object pose | `MujocoGtPoseSource` | `AprilTagPoseSource` |
| Base policy | scripted → BC (built + frozen here) | reused as-is |
| Intervention cost | initial `[250,200]` | re-tune |
| Camera calibration | n/a | intrinsics + hand-eye (only France-day asset task) |

**Everything except camera calibration and the device/backend swap is built and fully
validated in the lab.**

## 8. Testing / acceptance gates

1. **Env smoke test**: `FrankaEnv` connects to sim, `reset`/`step` run, obs shapes
   correct, `info['success']` toggles when cubes are stacked by hand-driven actions.
2. **Scripted policy**: completes stacking in sim above some success rate; mediocre
   variant fails at the precision step as intended.
   `scripts/franka_sim_grasp_demo.py` is a deliberately slow, human-reviewable mechanical
   acceptance video and should not define MILE collection timing; env-driven rollouts use
   `Franka-Stack-Sim-v0` through `FrankaEnv`.
3. **BC base policy**: trains, loads as `ActorCriticPolicy`, rolls out, is mediocre.
4. **Collector (headless)**: with `scripted_intervener`, produces a dataset whose dict
   keys/shapes exactly match `collect_synthetic_data`'s Box output; feeds `prepare_dataset`
   without error.
5. **Collector (teleop)**: SpaceMouse intervention recorded with correct ν and actions.
6. **End-to-end iterative run** in sim: N=2–3 rounds, k=3, success rate improves across
   rounds; `generate_rollout` guarded so no unwanted autonomous execution.
7. **Docker**: image builds; sim + collector + training run inside the container with
   SpaceMouse + GUI passthrough.

## 9. Risks & mitigations

- **rclpy ↔ pinned MILE deps coexistence** in one image → keep MILE off MetaWorld on the
  Franka path; pin and lock; test the image early.
- **Autonomous `generate_rollout` on hardware** → config-gated (Section 5.7); default off.
- **AprilTag occlusion during grasp** → only need pose pre-grasp; use grasp-transform
  after; mount tag clear of fingers.
- **Sim-to-real gap** in the base policy → keep it mediocre anyway (interventions are the
  point); state-based obs reduce the gap vs images.
- **France-day time** → robust warm-up behavior; rehearse calibration; everything else
  pre-validated in the lab.

## 10. Open items to confirm during planning
- Exact multipanda world-XML injection point for the cube assets.
- Control rate + delta scaling that feels right for SpaceMouse teleop.
- Whether to keep a 4-DoF down-facing wrist or expose yaw for stacking alignment.
