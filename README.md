# MILE: Model-based Intervention Learning
Codebase to replicate the results for https://liralab.usc.edu/mile/.

## Setting up the environment
- Create a new conda environment: `conda create -n mile python=3.10`
- Install required packages: `pip install -r requirements.txt` (this also installs the v2-compatible [Metaworld](https://github.com/Farama-Foundation/Metaworld) commit; do not `pip install metaworld`, as the PyPI release is v3 and requires `gymnasium>=1.1`, which is incompatible with this stack)
- Install MILE: `pip install -e .` (this also installs the `mile_franka` package used below)

## MILE on Franka — block-stacking tutorial

This fork adapts MILE to a Franka Panda block-stacking task ([design spec](docs/superpowers/specs/2026-06-15-mile-franka-stacking-design.md)). The whole pipeline runs **headless with a built-in fake backend** — no robot, no ROS, no MuJoCo, no GPU — so you can validate everything before touching hardware. The simulated/real Panda backends (multipanda_ros2 + SpaceMouse/Vive) drop into the same interfaces later.

Activate the environment first (`conda activate mile`). Run all commands from the repo root.

### 1. Smoke-test the environment

Drives the fake `FrankaEnv` through a full pick-and-place and checks `info['success']` fires:

```
python scripts/smoke_franka_env.py          # prints: smoke_franka_env ok
```

### 2. Build the mediocre base policy

Collects scripted demonstrations, BC-distills them into the `ActorCriticPolicy` MILE trains on, and reports the rollout success rate. The policy is deliberately *mediocre* (it starts the task but can't reliably finish) — that is the regime where human interventions matter.

```
python scripts/build_base_policy.py
# -> Saved base policy to trained_models/franka/base_policy; success rate = 0.57
```

A success rate roughly in `0.3–0.7` is what you want. If it solves everything (`1.00`) or nothing (`0.00`), tune the mediocre knobs in `ScriptedPolicyConfig` (`mile_franka/policies/scripted.py`): raise/lower `aim_xy_noise_std`, `release_height_error`, `action_noise_std`.

### 3. Run the iterative MILE loop (synthetic human)

`config_franka.json` runs MILE's iterative loop with a **scripted "human"** (`ScriptedIntervener`) standing in for the SpaceMouse/Vive operator, autonomous on-robot rollouts disabled (`rollout.auto_eval: false`). It loads the base policy from step 2.

```
cd scripts && python train_mile.py --config ../config_franka.json && cd ..
```

You should see `Round: 0` / `Round: 1`, the dataset growing each round, and the trained policy + mental model written to `output_dir/franka/`. Increase `num_rounds` / `episodes_per_round` / `train.num_epochs` in `config_franka.json` for a real run.

To use a **live SpaceMouse** instead of the scripted human, set `"intervener": "spacemouse"` in `config_franka.json` (requires `pyspacemouse` + the device attached).

### Multipanda MuJoCo sim path

The sim backend is registered as `Franka-Stack-Sim-v0` and uses the same `FrankaEnv` API as the fake backend, but with the multipanda stacking scene geometry (`cube_size=0.06`, `table_z=0.0`). Scripted policies and scripted interveners derive their task geometry from `env.unwrapped.config`, so fake-env and sim-env runs do not silently mix cube heights.

Use `scripts/franka_sim_grasp_demo.py` only as a slow mechanical/video acceptance check for the controller and gripper. For MILE data collection and training, use env-driven rollouts such as:

```
DISPLAY=:99 python scripts/franka_sim_rollout_record.py \
  --episodes 3 --out /tmp/rollout.mp4 --data /tmp/demos.npz
```

For a smoke run, tiny BC settings are useful for import/API validation, but they are not a quality check; use the normal `build_base_policy.py` defaults when you need a usable mediocre base policy.

### Toward the real robot

The real backend (hucebot's multipanda_ros2 controller docker), the MuJoCo cube assets, and the Vive/AprilTag swap are gated on hardware bring-up — see the [bring-up checklist](docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md) and resolve the `CONFIRM@bringup:` markers (`grep -rn CONFIRM@bringup mile_franka`).

---

The remaining sections document the **upstream MILE workflow on MetaWorld** (used for the Peg-Insert reproduction).

## Dataset generation
You can generate a synthetic dataset of interventions using our intervention model if you have a trained agent and mental model.

```
python scripts/collect_synthetic_interventions.py \
--env_name 'peg-insert-side-v2' \
--n_episodes 20 \
--rollout_policy 'path_to_your_rollout_policy' \
--intervention_policy 'path_to_expert_policy' \
--mental_model 'path_to_trained_mental_model' \
--save_path 'path_to_save' 
```

In order to pretrain the agent and the mental model, you can follow [SB3](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html) and [Imitation](https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html) documents.

## Training MILE
```
python scripts/train_mile.py --config 'config.json'
```

To reproduce results for Peg-Insert environment, download the pretrained models (using this drive [link](https://drive.google.com/file/d/1bzKGyOmX1ZCmAWnZiq_sAFRxi3AXvm4t/view?usp=drive_link) or via terminal) and extract the downloaded .zip file. 

```
gdown 1bzKGyOmX1ZCmAWnZiq_sAFRxi3AXvm4t 
unzip trained_models.zip
```

Then run `train_mile.py` with the default `config.json` file. 

## Evaluating MILE
```
python scripts/eval_mile.py --trained_model 'path_to_your_trained_model_dir' --num_episodes 100
```
