# MILE: Model-based Intervention Learning
Codebase to replicate the results for https://liralab.usc.edu/mile/.

## Setting up the environment
- Create a new conda environment: `conda create -n mile python=3.10`
- Install required packages: `pip install -r requirements.txt` (this also installs the v2-compatible [Metaworld](https://github.com/Farama-Foundation/Metaworld) commit; do not `pip install metaworld`, as the PyPI release is v3 and requires `gymnasium>=1.1`, which is incompatible with this stack)
- Install MILE: `pip install -e .` (this also installs the `mile_franka` package used below)

## MILE on Franka — block-stacking tutorial

> **Summer-school participants:** start at [docs/tutorial/00-setup.md](docs/tutorial/00-setup.md).

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

To put a **real human** in the loop instead of the scripted one, set `"intervener"` in `config_franka.json` to `"spacemouse"` (3Dconnexion SpaceMouse) or `"joystick"` (Xbox/PS gamepad), with the device attached. The full live-sim runbook for this is below ([Docker sim path — cold start](#docker-sim-path--cold-start-live-robot--human-teleop)).

### Multipanda MuJoCo sim path

The sim backend is registered as `Franka-Stack-Sim-v0` and uses the same `FrankaEnv` API as the fake backend, but with the multipanda stacking scene geometry (`cube_size=0.06`, `table_z=0.0`). Scripted policies and scripted interveners derive their task geometry from `env.unwrapped.config`, so fake-env and sim-env runs do not silently mix cube heights.

Use `scripts/franka_sim_grasp_demo.py` only as a slow mechanical/video acceptance check for the controller and gripper. For MILE data collection and training, use env-driven rollouts such as:

```
DISPLAY=:99 python scripts/franka_sim_rollout_record.py \
  --episodes 3 --out /tmp/rollout.mp4 --data /tmp/demos.npz
```

For a smoke run, tiny BC settings are useful for import/API validation, but they are not a quality check; use the normal `build_base_policy.py` defaults when you need a usable mediocre base policy.

### Docker sim path — cold start (live robot + human teleop)

Everything above runs headless on the host conda env. To run the **full human-in-the-loop loop on the multipanda MuJoCo sim** — a live MuJoCo window plus a real SpaceMouse or gamepad — use the Docker image and the `make` verbs ([docker design](docs/superpowers/specs/2026-06-16-mile-docker-image-and-sim-demos-design.md), [phase-a design](docs/superpowers/specs/2026-06-17-phase-a-sim-spacemouse-design.md)). Run everything from the repo root.

**Prerequisites:**
- Docker Engine + the Compose plugin — check with `docker compose version`. If `make build` prints `docker: No such file or directory`, Docker isn't on your shell's PATH: install it, or — if you just installed it — open a fresh shell (in zsh, `hash -r` clears the stale command cache).
- **The `hucebot:franka-humble` base image (build once per machine).** Our image layers the MILE stack on top of hucebot's multipanda_ros2 controller image, which is **not** on any registry — you build it from source. On a fresh machine, before `make build`:
  ```bash
  git clone https://github.com/hucebot/multipanda_ros2
  cd multipanda_ros2 && docker compose build   # produces image hucebot:franka-humble
  cd -
  ```
  `make build` then uses the classic Docker builder (`DOCKER_BUILDKIT=0`) on purpose: BuildKit would try to *pull* this local-only base from Docker Hub and fail with `pull access denied … hucebot:franka-humble`. Confirm the base is present with `docker images | grep hucebot`.
- For the live window: an X server on the host, with the container allowed to use it. Once per login: `xhost +local:root`.
- GPU is optional. `gpus: all` in `docker/docker-compose.yml` gives hardware GL + faster training; CPU-only works on software GL (~14 FPS).
- A teleop device for the human-in-the-loop step: a 3Dconnexion SpaceMouse **or** an Xbox/PS gamepad. Uncomment its device line in `docker/docker-compose.yml` — `/dev/hidraw0` for the SpaceMouse, `/dev/input/js0` + `/dev/input/event*` for the gamepad (verify the node with `ls -l /dev/input/js* /dev/hidraw*`).
- For the real-camera path: an Intel RealSense D415 (serial 217222067236). The docker-compose already lists the video nodes and USB passthrough — verify with `lsusb | grep RealSense` and check that the listed `/dev/video*` minors exist (`ls -l /dev/video*`).

**Steps:**

```bash
# 1. Build the image and start the persistent `sim` service
make build
make up

# 2. Launch the multipanda stacking sim (headless is fine for collection/training)
make sim-up            # headless; or `make sim-gui` for a live window (needs xhost)

# 3. Collect mediocre demos and BC-train the base policy
make collect-mediocre  # -> output_dir/franka/sim_demos_mediocre.npz
make base-policy       # BC-trains the mediocre base policy from those demos
make eval-base         # (optional) measure the base policy's sim success rate
make pose-test         # run 20 pose-layer unit tests (no ROS/hardware needed)

# --- Real FR3 path (hardware needed) ---
make apriltag-up       # launch D415 + AprilTag detection + calibration static tf
make calibrate-camera  # eye-to-hand calibration capture → camera_calib.yaml
make mile-real         # iterative MILE on the real FR3
make eval-real         # policy eval on the real FR3

# 4. Verify your teleop device reads inside the container
make spacemouse-check  # SpaceMouse: push the puck, expect nonzero deflection
#   or: make joystick-check   # gamepad: push the sticks / press buttons

# 5. Pick the human device and run the iterative MILE loop
#    edit config_franka.json -> "intervener": "spacemouse" | "joystick"
make sim-gui           # live window so you can see when to take over
make mile              # human overrides when the policy is about to fail; repeats N rounds
```

`make shell` opens an in-container shell with the env sourced; `make down` stops the service. `make mile` runs against a live sim, so keep `make sim-up`/`make sim-gui` running in another terminal.

### Toward the real robot

The real backend (hucebot's multipanda_ros2 controller docker), the MuJoCo cube assets, and the Vive/AprilTag swap are gated on hardware bring-up — see the [bring-up checklist](docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md) and the [status & TODOs](docs/superpowers/notes/2026-06-17-status-and-todos.md). Resolve the remaining `CONFIRM@bringup:` markers (`grep -rn CONFIRM@bringup mile_franka`).

**AprilTag pose pipeline — DONE (2026-06-18):** RealSense D415 color-only 640x480@15Hz, apriltag_ros tag36h11 detection, tf2 chain `panda_link0 → tag36h11:0` resolving, `AprilTagPoseSource` returning cube-center poses with <0.2mm stability. `make pose-test` runs 11 unit tests with no ROS/hardware needed. `make apriltag-up` launches the camera + detection inside the container. Docker passthrough for USB + video devices configured in `docker/docker-compose.yml`.

**Remaining for real MILE:** camera calibration (`calibrate_camera.py`), controller bring-up confirmation (FR3 + hucebot controller), real env builder (`Franka-Stack-Real-v0`), real config (`config_franka_real.json`), real eval script. Full details in the [status doc](docs/superpowers/notes/2026-06-17-status-and-todos.md) §Phase (b).

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
