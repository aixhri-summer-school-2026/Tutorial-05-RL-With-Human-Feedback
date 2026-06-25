# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository rules

- **No Claude/AI attribution in git commits** — omit the `Co-Authored-By: Claude`
  trailer and any "Generated with Claude Code" lines.
- **Dependency stack is tightly pinned and must not be casually bumped.** This code
  targets `gymnasium==0.29.1`, `stable-baselines3<=2.3.2`, `imitation`, and a **MetaWorld
  v2-era commit** (`mujoco<3`). Do **not** `pip install metaworld` (PyPI is v3 / needs
  `gymnasium>=1.1`, which breaks the stack). Install via `requirements.txt`.
- Python 3.10 (conda env).

## Goals

Upstream MILE is a research codebase for **Model-based Intervention Learning** (paper:
https://liralab.usc.edu/mile/). A human oversees a robot, intervenes only when it is
about to fail, and MILE learns from both intervention and *non*-intervention timesteps
by modeling *when/why* a human intervenes.

This fork's active project: **adapt MILE to run on a real Franka Panda (via hucebot's
`multipanda_ros2`) for a summer-school tutorial, using a block-stacking task.** The
design is built and validated against `multipanda_ros2`'s MuJoCo sim with an Xbox gamepad
(joystick) in the home lab, then deployed on hardware with an HTC Vive in France — the two backends run
the *same* Cartesian-impedance controller, so only the teleop device, object-pose source,
and robot backend swap. The design and verified codebase facts for the Franka adaptation
are recorded in the architecture section below.

## Setup

```bash
conda create -n mile python=3.10
pip install -r requirements.txt   # also installs the pinned MetaWorld commit
pip install -e .
```

Pretrained models for the Peg-Insert reproduction: `gdown 1bzKGyOmX1ZCmAWnZiq_sAFRxi3AXvm4t && unzip trained_models.zip`.

## Commands

```bash
# --- Upstream MILE on MetaWorld ---
# Train (offline or iterative — selected by config["experiment"]["mode"])
python scripts/train_mile.py --config config/metaworld.yaml

# Evaluate a trained model
python scripts/eval_mile.py --trained_model <dir> --num_episodes 100

# Generate a synthetic intervention dataset (needs trained rollout/expert/mental models)
python scripts/collect_synthetic_interventions.py \
  --env_name peg-insert-side-v2 --n_episodes 20 \
  --rollout_policy <p> --intervention_policy <p> --mental_model <p> --save_path <p>

# --- Franka block-stacking (single sim path, all in the MILE-on-multipanda image) ---
make build && make up         # build the image, start the persistent `sim` service
make sim-up                   # launch the multipanda MuJoCo stacking sim headless
make collect-mediocre         # MEDIOCRE rollouts -> sim_demos_mediocre.npz (feeds the base policy)
make collect-expert           # PERFECT successful-only rollouts -> sim_demos_expert.npz (reference)
make base-policy              # BC-train the mediocre base policy from sim_demos_mediocre.npz
make mile                     # iterative MILE run (config/franka_sim.yaml, Franka-Stack-Sim-v0)
make shell                    # interactive in-container shell (env sourced)
make joystick-check           # print live gamepad axes/buttons (sanity check)
make pose-test                # run 20 pose-layer unit tests (no ROS/hardware needed)
make franka-up               # launch the real FR3 controller stack (separate franka_ros2 container, CycloneDDS); ROBOT_IP/LOAD_GRIPPER overridable
make franka-shell            # shell in the franka_ros2 container (CycloneDDS sourced)
make apriltag-up              # (in-container) launch D415 + apriltag_ros + calibration static tf
make calibrate-camera         # (in-container) eye-to-hand calibration capture -> camera_calib.yaml
make close-gripper            # (in-container) clamp the gripper (e.g. onto the calib board); GRIP_FORCE/CLOSE_WIDTH overridable
make open-gripper             # (in-container) release the gripper; OPEN_WIDTH overridable
make mile-real                # (in-container) iterative MILE on real FR3 (config/franka_real.yaml)
make eval-real                # (in-container) policy eval on real FR3
make view-twin                # (in-container, host display) read-only MuJoCo twin: cubes from AprilTag, arm from /joint_states; safe to run alongside mile-real/eval-real
```

The `config/franka_sim.yaml` targets `Franka-Stack-Sim-v0` and runs in-container via `make mile`
against a live `make sim-up`: `mode: iterative`, `collector: real` + `intervener: joystick`
(human-in-the-loop with an Xbox gamepad), `rollout.auto_eval: false`,
`num_rounds: 5`, `episodes_per_round: 3`, `num_epochs: 500`. The fake env (`Franka-Stack-Fake-v0`)
is retained only for import-level checks, not as a pipeline gate. `config/metaworld.yaml` /
`config/franka_sim.yaml` are the single source of run configuration (env, modes,
policy/mental-model types and paths, logging, save, rollout, **`collector`**,
**`intervener`**, **`rollout.auto_eval`**).

There is a **pose-layer test suite**: `make pose-test` runs 20 unit tests (calibration,
AprilTag offset, AprilTagPoseSource, RosPoseStampedSource, hand-eye solve) with no
ROS/hardware needed. Run it after touching the pose layer.

Cubes are **5 cm everywhere** (`cube_size=0.05`: fake-env default, sim, and real). For
`Franka-Stack-Sim-v0` the table geometry still differs from the headless fake env
(`table_z=0.0`, matching the multipanda stacking scene; fake-env default `table_z=0.02`).
Scripted policies/interveners must take `env.unwrapped.config`; otherwise they silently
target the fake-env heights. `train_mile.py`, `build_base_policy.py`, and
`franka_sim_rollout_record.py` follow this rule.

## Architecture (big picture)

The system jointly trains two networks from a dataset of interventions, then deploys only
the policy.

- **`mile/computational_model.py`** — the heart of the method. `computational_intervention_model`
  implements the probit intervention model from the paper for both `Discrete` and `Box`
  action spaces, returning the intervention probability `p(ν=1|s)` and the final action
  distribution. `COST_LOOKUP` maps each env name to `[cost, cdf_scale]` hyperparameters;
  **every env used in training/collection must have an entry here or the trainer raises.**
  The Franka task has entries for both `Franka-Stack-Fake-v0` and `Franka-Stack-Sim-v0`
  (both `[70, 100.0]`, retuned from the `pick-place-v2` seed `[250, 200.0]` to reflect a
  lower intervention threshold; still to be calibrated against observed human rate on hardware).
- **`mile/algorithm.py`** — `InterventionTrainer` jointly optimizes the **policy** `π_θ`
  and the **mental model** `π̃_ξ` (what the human believes the robot will do). Continuous
  loss = BCE on the intervention flag ν **+** Gaussian NLL of the human action *only on
  ν=1 steps* (`mile_cont_loss_fn`). `generate_rollout` runs the policy on the env to
  measure success rate; it **is now guarded behind `experiment.rollout.auto_eval`** (both
  the `__init__` call and the in-loop calls), default `True` for MetaWorld, set `false` on
  the Franka/real path so the policy is never executed autonomously.
- **`scripts/train_mile.py`** — two modes. `offline`: train once on a fixed dataset.
  `iterative`: repeat `num_rounds` × {collect data → retrain}. The data collector is
  selected by `experiment.collector`: `synthetic` (default) uses `collect_synthetic_data`
  (a fully-synthetic "human" — `rollout_policy` + `computational_intervention_model` +
  `gt_mental_model` + expert `intervention_policy`); `real` uses
  `mile_franka.collect.Collector` with a real human-in-the-loop intervener. MetaWorld is
  imported lazily, so the Franka path does not require it.
- **`scripts/collect_synthetic_interventions.py`** — the synthetic data collector used by
  iterative training.
- **`mile/utils.py`** — `prepare_dataset` (train/val split; **stacks all dict keys with
  `np.array`, so every key must be array-convertible with consistent shape**),
  `DictDataset`, config IO, and the `Logger` wrapping imitation's hierarchical logger
  (stdout / tensorboard / wandb).

### Dataset schema (continuous / `Box`)

The collector and trainer agree on this dict: `state, rollout_action, action,
intervention_prob, intervention, reward, next_state, done`. The trainer's continuous loss
uses only `state`, `action` (= human action when intervening), and `intervention` (ν).
**`intervention_prob` is read but immediately overwritten** by the intervention model, so
a real human-in-the-loop collector can fill it with a placeholder.

### Observation / action conventions

MetaWorld envs are wrapped with `FrameStack(4)` then `FlattenObservation` to retain
temporal context; the action space is 4-DoF (xyz end-effector delta + gripper), EE
pointing down. Policies/mental models are `ActorCriticPolicy` (`bc`) or `QNetwork`/SAC,
net `[256, 256]`, with `NormalizeFeaturesExtractor` + `RunningNorm`.

## The `mile_franka` package (Franka block-stacking)

A separate package alongside `mile/` implementing the Franka adaptation. Backend-agnostic:
the same `FrankaEnv` runs against a no-ROS fake backend (dev/CI) and the real
multipanda_ros2 controller (sim/hardware) — only the backend and pose source swap. ROS
deps (`rclpy`, `franka_msgs`, `geometry_msgs`) are imported **lazily**, so the package
imports and the smoke scripts run with no ROS installed.

- **`mile_franka/config.py`** — `StackTaskConfig` (cube/workspace/threshold dims shared by
  sim and real) + `DOWN_QUAT`.
- **`mile_franka/envs/`** — `RobotBackend` ABC (`backend.py`); `FakeWorld`/`FakeRobotBackend`/
  `WorldPoseSource` (`fake_backend.py`, a kinematic pick-and-place world for headless runs);
  `FrankaEnv` (`franka_env.py`, gym env: delta→Cartesian target; reduced 9-dim obs
  `[ee_xyz, gripper_width, top_xyz, bottom_xy]` (quats + bottom_z dropped; `--include_bottom_z`
  adds bottom_z → 10-dim) → `(90,)` after `FrameStack(10)+Flatten`; `privileged_frame()`
  returns the full 18-dim GT frame for scripted planners; real obs z is mapped into the training
  table frame by `TablePlaneCanonicalizer`; `info['success']`); `registration.py`
  (`register_franka_envs()`, `make_franka_env()`, `FRANKA_FRAME_STACK=10`,
  `FAKE_ENV_ID="Franka-Stack-Fake-v0"`); `ros_backend.py` (`MultipandaRosBackend`, real,
  **bring-up-gated** — joins hucebot's controller docker DDS graph).
- **`mile_franka/pose/`** — `Pose`/`ObjectPoseSource` ABCs (`base.py`);
  `AprilTagPoseSource` (`apriltag.py`, tf2-backed, primary real-cube source — live-verified
  with D415 on 2026-06-18); `RosPoseStampedSource` (`ros_posestamped.py`, generic
  FoundationPose-ready seam); `CameraCalibration` (`calibration.py`, camera→base extrinsics
  file format + YAML load + static-transform args); `MujocoGtPoseSource` (`mujoco_gt.py`,
  sim-only); `CanonicalCubePoseSource`/`TablePlaneCanonicalizer` (`canonical.py`, real-only:
  the former stamps the training cube quaternion, the latter measures the table height from the
  resting cube and shifts `ee_z`/`top_z` into the training table frame — the lifted cube is
  tracked continuously, never snapped); `cube_center_pose()` pure half-edge offset helper.
- **`mile_franka/teleop/`** — `TeleopDevice`/`TeleopReading` ABCs (`base.py`);
  `JoystickDevice` (`joystick.py`, pygame gamepad — **the default dev teleop device**, ν via a
  stateful clutch-toggle segment) and `SpaceMouseDevice` (`spacemouse.py`, kept as an alternative,
  ν = clutch-or-motion) — both use an injectable raw reader so they load/test without hardware.
  Select via `intervener: joystick|spacemouse` in `config/franka_sim.yaml` (currently `joystick`).
- **`mile_franka/policies/`** — `ScriptedStackPolicy` (`scripted.py`, mediocre state machine
  over GT poses — reads `env.unwrapped.privileged_frame()`, not the reduced obs; per-episode
  randomizes transit/lift height U[0.10,0.25] and placement gap U[0.002,0.015]; pass the active
  env's `StackTaskConfig` for sim/hardware); `build_actor_critic_policy`/`train_bc` (`bc.py`,
  MILE-arch policy + imitation BC, plain `NormalizeFeaturesExtractor`+RunningNorm — the reduced
  obs already drops the dead dims, so no masking); `collect_scripted_demos` (`demos.py`).
- **`mile_franka/collect.py`** — `Collector` (replaces `collect_synthetic_data`; returns the
  same `(dataset, mean_score, mean_success)` tuple) + `ScriptedIntervener` (headless scripted
  "human") / `TeleopIntervener` (adapts any `TeleopDevice`). Emits the Box dataset via
  `InterventionDatasetBuilder` (`data/dataset.py`, `BOX_KEYS` is the single schema source).
- **`mile_franka/envs/ros_backend.py` + `pose/mujoco_gt.py` carry `CONFIRM@bringup:` markers**
  (`grep -rn CONFIRM@bringup mile_franka`) for topic/action/frame names to confirm against the
  live hucebot controller; see `docs/superpowers/bringup-reference.md`.

The real backend must run against **hucebot's multipanda_ros2 controller docker** (the France lab's image), not a hand-rolled stack.
