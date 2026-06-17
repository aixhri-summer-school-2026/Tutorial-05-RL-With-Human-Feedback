# MILE-on-Multipanda Docker Image + Sim Demo Collection / Base-Policy Training

**Date:** 2026-06-16
**Author:** rayray2002
**Status:** Design (approved — ready for implementation plan)

Increment of the parent spec `2026-06-15-mile-franka-stacking-design.md` (§6 Docker
packaging, §5.5 base policy, §8 gates 2–3). This narrows §6 from "we will ship one image"
to a concrete, buildable image + ergonomics, and resolves how the **base BC policy** is
collected and trained from the **live multipanda MuJoCo sim** rather than the fake env.

## 1. Goal

1. Build the **single MILE-on-multipanda Docker image** the parent spec calls for, so the
   whole stack runs **fully in-container** (collection *and* training) and ships to France
   reproducibly.
2. Make driving the container ergonomic — replace raw `docker exec … cd … PYTHONPATH …`
   with short verbs.
3. Use that image to **collect scripted sim demos through `FrankaEnv`** and **BC-train the
   (deliberately mediocre) base policy** in the same obs space it will be deployed in.
4. Shape the image, mounts, Makefile, config layout, and base-policy obs space so the
   **iterative MILE run in sim drops in with no rework** (no reverting `config_franka.json`,
   no second image, no re-trained base policy). See §7.

## 2. Why fully in-container (decision)

The current container `hucebot:franka-humble` runs the sim + `rclpy` + `mile_franka` but
has **no `torch`/`imitation`/`stable-baselines3`**. That forced a temporary split: collect
in-container, BC-train on the host conda env. That split is rejected here because it keeps
two environments in sync and does not ship to France. Adding the MILE layer to the image
collapses collect+train into one environment **and** produces the France-shippable
artifact, so no work is throwaway.

## 3. Architecture

### 3.1 The image (`docker/Dockerfile`)

- `FROM hucebot:franka-humble` (ROS 2 Humble, Ubuntu 22.04, Python 3.10, MuJoCo 3.2.0
  C++-only, `rclpy`, `gymnasium==0.29.1`, `numpy 2.2.6` already present).
- pip-install the pinned MILE stack **without disturbing** the base `gymnasium==0.29.1`:
  `torch`, `imitation`, `stable-baselines3<=2.3.2`, `wandb`, `tensorboard`. Include
  `pyspacemouse`/`hidapi` now (lab SpaceMouse) and an AprilTag lib now (used in France,
  harmless in the lab) so France-day is config-only, not a rebuild (§7). The Vive path adds
  no Python dep — it is an `rclpy` subscriber on the base image.
- **MetaWorld-free.** MetaWorld is the only thing needing the `mujoco<3` *Python* binding;
  the sim's MuJoCo is C++-only, so excluding MetaWorld means zero MuJoCo-Python conflict
  (see `franka-sim-verified-bringup`). Keep MetaWorld an optional extra, never installed on
  this image.
- `pip install -e .` for the `mile` + `mile_franka` packages.
- Produce a **locked** requirements file for reproducibility.
- **Dev vs ship:** keep the repo **bind-mounted** during lab dev (edit on host, run
  in-container, no rebuild per code change). Bake the repo into the image only for the
  final France ship.

### 3.2 Orchestration (`docker/docker-compose.yml`)

One `sim` service:
- bind mount `..:/home/user/mile-code` (RW) — matches today's mount; host edits live.
- `network_mode: host` (or a shared `ROS_DOMAIN_ID`) for the DDS graph.
- X11 passthrough (`DISPLAY`, `/tmp/.X11-unix`) for the headless GLFW viewer + ffmpeg.
- `/dev/hidraw*` passthrough (lab SpaceMouse) and `/dev/video*` / RealSense passthrough
  (France camera for AprilTag) — both declared now (commented/optional is fine), harmless
  in the lab sim, so France-day is config-only (§7).
- optional `gpus: all` (MLP trains fine on CPU; GPU optional).
- a named volume / bound `output_dir/` for saved models + datasets.
- Replaces the current ad-hoc `sleep infinity` container; `command: sleep infinity` keeps
  it a persistent box.

### 3.3 Ergonomics (`Makefile` at repo root)

Thin verbs wrapping `docker compose exec sim bash -lc '<recipe>'`, each encapsulating the
verified headless recipe (LD_LIBRARY_PATH, ROS sourcing, `RMW_IMPLEMENTATION`, Xvfb `:99`,
software GL) from `franka-sim-verified-bringup`:

- `make build` — build the image.
- `make up` / `make down` — start/stop the compose service.
- `make sim-up` — Xvfb `:99` + `ros2 launch …franka_sim_stacking.launch.py` (cubes +
  inactive controllers), run detached (`docker exec -d` semantics so it survives).
- `make collect` — `scripts/franka_sim_rollout_record.py` (scripted, mediocre) → MP4 +
  demo `.npz` under `output_dir/franka/`.
- `make base-policy` — BC-train from the collected `.npz` (now in-container).
- `make mile` — the iterative MILE run against the live sim: `cd scripts && python
  train_mile.py --config ../config_franka_sim.json` (requires `make sim-up`; see §7).
- `make shell` — interactive shell, pre-`cd`'d into the repo with `PYTHONPATH`/ROS env set.

Compose is the reproducibility substrate; the Makefile is the one-command UX. Both, not
either/or. The two-container DDS split is explicitly **not** chosen (parent spec §6).

## 4. Build order & the dependency-coexistence spike (do first)

The parent spec's top risk (§9) is **rclpy ↔ pinned MILE deps coexistence in one image**.
Retire it before writing the real Dockerfile:

1. **Dep spike** — in a scratch layer / throwaway `docker commit`, pip-install the MILE
   stack on top of the base image and confirm **all coexist**:
   `python3 -c "import rclpy, torch, stable_baselines3, imitation, gymnasium, mile_franka"`
   and `gymnasium.__version__ == 0.29.1` still holds. If `numpy 2.x` fights `sb3<=2.3.2`,
   pin numpy down and re-verify the sim (`rclpy` / `mujoco_ros_msgs`) still imports.
2. Only then write `docker/Dockerfile` + lockfile from the spike's known-good set.
3. `docker/docker-compose.yml`.
4. `Makefile` verbs.

## 5. Sim demo collection & base policy

Demo source = **multipanda sim via `FrankaEnv` (`Franka-Stack-Sim-v0`)**, chosen so the BC
base policy is trained in the **same observation space it is deployed in** (sim EE obs is
the `panda_hand` body pose, ≈10 cm above the TCP — distinct from the fake env). The policy
is **deliberately mediocre** (`mediocre=True` scripted demos), per parent spec §5.5.

### 5.1 Feasibility gate (open item from the bring-up notes)

`scripts/franka_sim_grasp_demo.py` (slow min-jerk waypoints) stacks in sim, but scripted
rollouts **through `FrankaEnv.step` (10 Hz delta actions)** are **not yet confirmed** to
descend to cube-center and grasp (`2026-06-15-phase2-bringup-checklist.md`: "the MILE
delta-action policy must descend to cube-center using the same gains"). Likely culprits to
check: the `panda_hand`-vs-TCP ~10 cm obs offset (`EE_BODY = panda_hand` in
`ros_backend.py`) and per-step settle time vs `SIM_STACKING_GAINS`.

**Gate:** `make collect` must yield **coherent** rollouts — watch the recorded MP4: the arm
approaches the cube, descends, closes, lifts, moves over the base, releases. It need not
succeed often (mediocre by design), but it must not flail. If incoherent, tune
`ros_backend.py` (TCP offset and/or gains/settle) until coherent **before** trusting the
demos.

### 5.2 Collect → train

- **Collect:** `franka_sim_rollout_record.py` already runs `ScriptedStackPolicy` over
  `FrankaEnv` and saves `obs / acts / episode` to `.npz` (light deps, runs in-container).
- **Train (gap to fill):** `build_base_policy.py` today collects+trains in one process on
  the **fake** env and cannot read an `.npz`. Add an **npz-fed BC path** (either a
  `--demos <npz>` flag on `build_base_policy.py` or a small `train_bc_from_npz.py`) that
  loads the saved sim transitions, BC-trains an `ActorCriticPolicy`
  (`FrameStack(4)+Flatten` obs, net `[256,256]`), saves to
  `trained_models/franka/base_policy`, and reports rollout success. Runs in-container now
  that torch is present.

## 6. Acceptance gates

1. **Dep spike** — MILE stack + rclpy + sim coexist in the image; `gymnasium==0.29.1`
   preserved; sim still launches.
2. **Image + compose + Makefile** — `make build` succeeds; `make up && make shell` lands
   in a ready shell; `make sim-up` brings the stacking sim up headless.
3. **Collection coherence** — `make collect` produces a coherent rollout MP4 + a `.npz`
   whose `obs` shape matches `FrameStack(4)+Flatten` of `Franka-Stack-Sim-v0`.
4. **Base policy** — `make base-policy` trains from that `.npz`, loads as
   `ActorCriticPolicy`, rolls out, is mediocre (low but non-zero success).
5. **MILE-readiness smoke** — `make mile` runs ≥1 iterative round in the live sim fully
   in-container (scripted intervener), consuming the sim-trained base policy without error.

## 7. Forward compatibility with the iterative MILE run (no future reverts)

The iterative MILE run (parent spec §5.6–5.7) is the next consumer of this image. Its full
implementation is a later increment, but the decisions below are made **now** so nothing
here has to be reverted then. Verified against `scripts/train_mile.py`
(`build_franka_or_metaworld_env` → `make_franka_env` for `Franka-*`; `collector: real` →
`mile_franka.collect.Collector`; `rollout.auto_eval` guard) and `config_franka.json`.

- **One image carries the whole pipeline.** The MILE run needs only `torch` / `imitation` /
  `stable-baselines3` / `wandb` / `tensorboard` — all already in §3.1. No second build, no
  extra layer later. The Box-dataset collector (`mile_franka/data/dataset.py` `BOX_KEYS`)
  and `InterventionTrainer` are pure-Python on top of that stack.

- **Base-policy obs space is chosen *for* MILE, not just for BC.** The sim MILE
  `Collector` runs the base policy on `FrankaEnv` via `make_franka_env`
  (`FrameStack(4)+Flatten`). Training the base policy on **sim** demos (§5) means the policy
  the collector loads already lives in the deployment obs space — this is the reason for
  the sim-demo choice, not an afterthought. `config_franka_sim.json`'s `policy_path` reuses
  `../trained_models/franka/base_policy`.

- **Two configs, never reverted:** keep `config_franka.json` as the **fake-env headless
  CI gate** (parent spec acceptance gate 6) untouched. Add **`config_franka_sim.json`** for
  the in-docker sim run — identical except `env_name: Franka-Stack-Sim-v0` (and any tuned
  `num_rounds`/`episodes_per_round`). Both keep `collector: real`,
  `intervener: scripted`, `rollout.auto_eval: false`, and the same
  `policy_path`/`save.outdir` layout. The fake gate and the sim run therefore want
  **different base-policy artifacts in the same obs *shape* (72,) but different obs
  *semantics*** (fake EE ≈ TCP; sim EE = `panda_hand`). Resolve by giving the fake gate its
  own fake-trained policy path (e.g. `base_policy_fake`) so a sim-trained policy is never
  silently rolled out on the fake env. Decide the two paths now; do not share one artifact.

- **`cd scripts` + mounts:** `config_franka_sim.json` uses paths relative to `scripts/`
  (`../trained_models/...`, `../output_dir/franka`), so `make mile` must `cd scripts`
  (parent spec / CLAUDE.md). The repo bind mount already covers `trained_models/` and
  `output_dir/`; no separate volume wiring needed.

- **Live-sim dependency is shared.** Both `make collect` (base demos) and `make mile`
  (intervention collection) step `FrankaEnv` against the **same** live sim — one
  `make sim-up`. The Makefile documents/enforces that ordering so the MILE run does not
  need a different launch path.

- **Teleop progression needs no rework — scripted → SpaceMouse → Vive.** All three are the
  same `TeleopDevice`/`TeleopIntervener` interface behind `collector: real`, so each swap is
  config-only (parent spec §5.3). To make that true in the image:
  - **SpaceMouse (lab):** the image carries `pyspacemouse`/`hidapi`; compose declares
    `/dev/hidraw*` (§3.2). Headless `ScriptedIntervener` is the no-device default first.
  - **Vive (France):** `ViveDevice` is a thin `rclpy` subscriber to hucebot's
    `vive_controller` topics (`/vive/right/pose`, `/vive/right/joint_states`); needs no new
    Python deps beyond the base `rclpy`, and the clutch/trigger maps directly to ν. It
    reaches the driver over the same `network_mode: host` DDS graph already in §3.2 — no
    image change in France, only run the hucebot vive node alongside.

- **Real-robot path is a backend/pose swap, not an image change.** The same image runs on
  hardware: `FrankaEnv` + `MultipandaRosBackend(mode='real')` against the **same
  Cartesian-impedance controller** (the whole sim→hardware strategy), swapping only the
  `ObjectPoseSource` (MuJoCo GT → `AprilTagPoseSource`) and the operator-gated reset. So the
  image/compose declare the France-only passthroughs **now** (commented/optional is fine):
  `/dev/video*` / RealSense for the camera, an AprilTag lib in the pip set. `rollout.auto_eval`
  stays `false` so the policy is never autonomously executed on hardware. Nothing structural
  changes between the lab sim run and the France hardware run — only the teleop device,
  pose source, and the launch target the backend connects to.

- **The two dataset formats stay distinct and both run in-container:** BC demos are
  `obs/acts` `.npz` (§5.2); the MILE dataset is the `BOX_KEYS` dict built by the
  `Collector`. Neither needs the host.

## 8. Out of scope (this increment)

- The full iterative MILE implementation/tuning (parent spec §5.6–5.7) — only the §7
  forward-compat constraints and the `make mile` MILE-readiness smoke (gate 5) are in scope.
- The **driver/wiring code** for SpaceMouse/Vive teleop and the AprilTag pose source, and
  GPU-required FoundationPose. (The image deps + compose passthroughs that *enable* them are
  in scope per §3 and §7 — only the device-integration code is deferred.)
- France-day camera calibration.
- Baking the repo into the image for shipping (done at ship time, not in lab dev).
