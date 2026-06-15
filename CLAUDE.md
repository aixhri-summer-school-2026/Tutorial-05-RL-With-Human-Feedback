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
design is built and validated against `multipanda_ros2`'s MuJoCo sim with a SpaceMouse in
the home lab, then deployed on hardware with an HTC Vive in France — the two backends run
the *same* Cartesian-impedance controller, so only the teleop device, object-pose source,
and robot backend swap. **Read `docs/superpowers/specs/2026-06-15-mile-franka-stacking-design.md`
before working on the Franka adaptation** — it records the design and the verified
codebase facts below.

## Setup

```bash
conda create -n mile python=3.10
pip install -r requirements.txt   # also installs the pinned MetaWorld commit
pip install -e .
```

Pretrained models for the Peg-Insert reproduction: `gdown 1bzKGyOmX1ZCmAWnZiq_sAFRxi3AXvm4t && unzip trained_models.zip`.

## Commands

```bash
# Train (offline or iterative — selected by config["experiment"]["mode"])
python scripts/train_mile.py --config config.json

# Evaluate a trained model
python scripts/eval_mile.py --trained_model <dir> --num_episodes 100

# Generate a synthetic intervention dataset (needs trained rollout/expert/mental models)
python scripts/collect_synthetic_interventions.py \
  --env_name peg-insert-side-v2 --n_episodes 20 \
  --rollout_policy <p> --intervention_policy <p> --mental_model <p> --save_path <p>
```

There is no test suite, linter, or build step. `config.json` is the single source of run
configuration (env, modes, policy/mental-model types and paths, logging, save, rollout).

## Architecture (big picture)

The system jointly trains two networks from a dataset of interventions, then deploys only
the policy.

- **`mile/computational_model.py`** — the heart of the method. `computational_intervention_model`
  implements the probit intervention model from the paper for both `Discrete` and `Box`
  action spaces, returning the intervention probability `p(ν=1|s)` and the final action
  distribution. `COST_LOOKUP` maps each env name to `[cost, cdf_scale]` hyperparameters;
  **every env used in training/collection must have an entry here or the trainer raises.**
- **`mile/algorithm.py`** — `InterventionTrainer` jointly optimizes the **policy** `π_θ`
  and the **mental model** `π̃_ξ` (what the human believes the robot will do). Continuous
  loss = BCE on the intervention flag ν **+** Gaussian NLL of the human action *only on
  ν=1 steps* (`mile_cont_loss_fn`). `generate_rollout` runs the policy on the env to
  measure success rate and **is called unconditionally in `__init__`** (≈line 186) — on a
  real robot this autonomously executes the policy, so it must be guarded.
- **`scripts/train_mile.py`** — two modes. `offline`: train once on a fixed dataset.
  `iterative`: repeat `num_rounds` × {collect data → retrain}. **In the current code the
  "human" is fully synthetic** — `collect_synthetic_data` rolls out a `rollout_policy`,
  decides interventions via the `computational_intervention_model` + a `gt_mental_model`,
  and takes actions from an expert `intervention_policy`. There is **no real
  human-in-the-loop teleoperation in this repo**; the Franka project adds it.
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
