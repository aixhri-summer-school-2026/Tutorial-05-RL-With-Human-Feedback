# MILE-on-Multipanda Docker Image + Sim Demos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the single MILE-on-multipanda Docker image (compose + Makefile) and use it, fully in-container, to collect scripted sim demos through `FrankaEnv` and BC-train the mediocre base policy — with the iterative MILE run dropping in unchanged.

**Architecture:** A thin image `FROM hucebot:franka-humble` adds the pinned MILE stack (torch/imitation/sb3) and `pip install -e .`, MetaWorld-free. `docker-compose.yml` defines one persistent `sim` service (host DDS, X11/Xvfb, device passthroughs); a root `Makefile` wraps the verified headless recipe into short verbs. `Franka-Stack-Sim-v0` is the single backend for collection, base-policy training, and the MILE run (the fake env is dropped as a path).

**Tech Stack:** Docker + docker-compose, ROS 2 Humble / multipanda MuJoCo sim, `gymnasium==0.29.1`, `stable-baselines3<=2.3.2`, `imitation`, `torch`, the `mile` + `mile_franka` packages.

**Spec:** `docs/superpowers/specs/2026-06-16-mile-docker-image-and-sim-demos-design.md`

**Convention note:** This repo has no pytest suite, linter, or build step (CLAUDE.md). The smoke scripts and in-container runs are the de-facto tests; verification steps below are container commands with expected output, not unit tests. No Claude/AI attribution in commit messages.

---

## File Structure

- Create `docker/requirements-mile.txt` — input pins for the MILE layer.
- Create `docker/requirements-mile.lock.txt` — frozen output of the dep spike (reproducibility).
- Create `docker/Dockerfile` — `FROM hucebot:franka-humble` + MILE layer.
- Create `docker/docker-compose.yml` — the `sim` service.
- Create `scripts/in_container_env.sh` — sourced env (ROS, LD_LIBRARY_PATH, RMW, headless GL).
- Create `scripts/sim_up.sh` — inject cube scene + Xvfb + launch the stacking sim.
- Create `Makefile` (repo root) — `build/up/down/shell/sim-up/collect/base-policy/mile`.
- Modify `scripts/build_base_policy.py` — add `--demos <npz>` flag; default env = sim.
- Modify `config_franka.json` — `env_name` → `Franka-Stack-Sim-v0`.
- Modify `mile_franka/envs/ros_backend.py` — **only if** the coherence gate (Task 7) fails.
- Modify `CLAUDE.md` + `docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md` — docs.

---

## Task 1: Dependency-coexistence spike (retire the top risk first)

**Files:**
- Create: `docker/requirements-mile.txt`
- Create: `docker/requirements-mile.lock.txt`

Goal: prove the MILE stack pip-installs on top of `hucebot:franka-humble` without breaking the base `gymnasium==0.29.1` / `rclpy` / `mujoco_ros_msgs`, and capture exact pins.

- [ ] **Step 1: Write the input requirements file**

Create `docker/requirements-mile.txt`:

```
# MILE training layer installed on top of hucebot:franka-humble.
# Keep the base image's gymnasium==0.29.1 and rclpy intact. MetaWorld is intentionally absent.
gymnasium==0.29.1
stable-baselines3<=2.3.2
imitation
wandb
tensorboard
pyspacemouse
hidapi
pupil-apriltags
# torch is installed separately in the Dockerfile (CPU wheel) to control the index URL.
```

- [ ] **Step 2: Run the spike in a throwaway container**

Run:
```bash
docker run --rm -v /home/ray/mile-code:/home/user/mile-code hucebot:franka-humble \
  bash -lc '
    set -e
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    pip install --no-cache-dir -r /home/user/mile-code/docker/requirements-mile.txt
    cd /home/user/mile-code && pip install --no-cache-dir -e .
    python3 - <<PY
import gymnasium, rclpy, torch, stable_baselines3, imitation, mile, mile_franka
from mujoco_ros_msgs.srv import GetBodyState
assert gymnasium.__version__ == "0.29.1", gymnasium.__version__
print("COEXIST OK; gymnasium", gymnasium.__version__, "numpy", __import__("numpy").__version__)
PY
    pip freeze
  ' | tee /tmp/mile_dep_spike.txt
```
Expected: ends with `COEXIST OK; gymnasium 0.29.1 numpy <X>` and a full `pip freeze`.

- [ ] **Step 3: Handle a numpy/gymnasium conflict if it appears**

If the import block fails because `imitation`/`sb3` pulled `gymnasium>0.29.1` or a `numpy` that breaks `rclpy`/`mujoco_ros_msgs`: add the offending pin to `docker/requirements-mile.txt` (e.g. `numpy<2` and re-assert `gymnasium==0.29.1`), then re-run Step 2 until `COEXIST OK` prints **and** `from mujoco_ros_msgs.srv import GetBodyState` still imports.

- [ ] **Step 4: Capture the lock file**

Extract the resolved versions from the spike output into `docker/requirements-mile.lock.txt`:
```bash
sed -n '/^[A-Za-z0-9_.-]\+==/p' /tmp/mile_dep_spike.txt | sort -u > docker/requirements-mile.lock.txt
```
Open `docker/requirements-mile.lock.txt` and confirm it contains `gymnasium==0.29.1`, `stable-baselines3==…`, `imitation==…`, `torch==…`, `numpy==…`. This is the reproducibility artifact.

- [ ] **Step 5: Commit**

```bash
git add docker/requirements-mile.txt docker/requirements-mile.lock.txt
git commit -m "Add MILE Docker layer requirements + locked dep set (coexistence spike)"
```

---

## Task 2: Dockerfile + image build

**Files:**
- Create: `docker/Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

Create `docker/Dockerfile`:

```dockerfile
# Single MILE-on-multipanda image: hucebot base + the pinned MILE training stack.
# MetaWorld-free (the sim's MuJoCo is C++-only; no mujoco<3 Python binding needed).
FROM hucebot:franka-humble

# Install the MILE layer. CPU torch wheel keeps the image lean; the MLP trains on CPU.
# (For a CUDA box, swap the index URL and add `gpus: all` in docker-compose.yml.)
COPY docker/requirements-mile.txt /tmp/requirements-mile.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements-mile.txt

# Bake the repo for shipping; in lab dev a bind mount (docker-compose.yml) overlays the
# same path, so `pip install -e .` keeps working against live host edits.
COPY . /home/user/mile-code
WORKDIR /home/user/mile-code
RUN pip install --no-cache-dir -e .

# Fail the build early if the deps don't coexist.
RUN python3 -c "import rclpy, torch, stable_baselines3, imitation, gymnasium, mile, mile_franka; \
from mujoco_ros_msgs.srv import GetBodyState; \
assert gymnasium.__version__=='0.29.1', gymnasium.__version__; print('deps OK')"

CMD ["sleep", "infinity"]
```

- [ ] **Step 2: Build the image**

Run:
```bash
docker build -f docker/Dockerfile -t mile:franka-humble /home/ray/mile-code
```
Expected: build succeeds and prints `deps OK` near the end.

- [ ] **Step 3: Handle pip permission errors (only if Step 2 fails on write perms)**

If `pip install` fails with a permissions error, the base image runs as a non-root user. Add `USER root` immediately after the `FROM` line, rebuild, and confirm the editable install path `/home/user/mile-code` is writable by the runtime user (the bind mount in Task 3 is host-owned, which is fine for `-e .`).

- [ ] **Step 4: Commit**

```bash
git add docker/Dockerfile
git commit -m "Add MILE-on-multipanda Dockerfile (FROM hucebot base, build-time dep check)"
```

---

## Task 3: docker-compose service

**Files:**
- Create: `docker/docker-compose.yml`

- [ ] **Step 1: Write the compose file**

Create `docker/docker-compose.yml`:

```yaml
# One persistent MILE-on-multipanda service. `network_mode: host` shares the DDS graph with
# the sim; an internal Xvfb (:99) provides the headless display. Device/GPU passthroughs are
# declared but commented so France-day (SpaceMouse/camera/GPU) is config-only, not a rebuild.
services:
  sim:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: mile:franka-humble
    container_name: mile_sim
    network_mode: host
    working_dir: /home/user/mile-code
    environment:
      - DISPLAY=:99
      - LIBGL_ALWAYS_SOFTWARE=1
      - GALLIUM_DRIVER=llvmpipe
      - RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    volumes:
      - ..:/home/user/mile-code            # live host edits in dev
      - /tmp/.X11-unix:/tmp/.X11-unix      # optional: view from host X if desired
    # devices:
    #   - /dev/hidraw0:/dev/hidraw0        # SpaceMouse (lab) — uncomment when attached
    #   - /dev/video0:/dev/video0          # camera (France) — uncomment on hardware
    # gpus: all                            # optional CUDA torch
    command: sleep infinity
```

- [ ] **Step 2: Stop the old ad-hoc container and start the service**

Run:
```bash
docker rm -f mile_sim_smoke 2>/dev/null; true
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml ps
```
Expected: a running `mile_sim` service.

- [ ] **Step 3: Verify the MILE stack is importable inside the service**

Run:
```bash
docker compose -f docker/docker-compose.yml exec sim \
  python3 -c "import torch, imitation, mile_franka; print('in-container deps OK')"
```
Expected: `in-container deps OK`.

- [ ] **Step 4: Commit**

```bash
git add docker/docker-compose.yml
git commit -m "Add docker-compose sim service (host DDS, headless display, device passthroughs)"
```

---

## Task 4: In-container env helper + core Makefile verbs

**Files:**
- Create: `scripts/in_container_env.sh`
- Create: `Makefile`

- [ ] **Step 1: Write the sourced env helper**

Create `scripts/in_container_env.sh`:

```bash
#!/usr/bin/env bash
# Common in-container environment for the MILE/multipanda sim. `source` me before any verb.
# `bash -lc` does NOT source ~/.bashrc reliably, so set the sim's runtime deps explicitly
# (see the franka-sim-verified-bringup note).
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/user/Libraries/libfranka/lib:/home/user/Libraries/mujoco/lib"
source /opt/ros/humble/setup.bash
source /home/user/humble_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export DISPLAY="${DISPLAY:-:99}"
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
cd /home/user/mile-code
```

- [ ] **Step 2: Write the Makefile with the core verbs**

Create `Makefile` (repo root). Uses `DC` for the compose invocation and `RUN`/`RUND` to wrap exec with the env helper:

```makefile
# MILE-on-multipanda dev verbs. One short command per action; all run inside the `sim`
# compose service with the verified headless env sourced. See docs/superpowers/specs/
# 2026-06-16-mile-docker-image-and-sim-demos-design.md.
DC      := docker compose -f docker/docker-compose.yml
ENVSH   := source scripts/in_container_env.sh
RUN      = $(DC) exec sim bash -lc '$(ENVSH) && $(1)'
RUND     = $(DC) exec -d sim bash -lc '$(ENVSH) && $(1)'

.PHONY: build up down shell sim-up collect base-policy mile

build:                       ## build the image
	$(DC) build

up:                          ## start the persistent sim service
	$(DC) up -d

down:                        ## stop the service
	$(DC) down

shell:                       ## interactive shell, env sourced, cd'd into the repo
	$(DC) exec sim bash -lc '$(ENVSH) && exec bash'

sim-up:                      ## launch the stacking sim headless (detached)
	$(call RUND,bash scripts/sim_up.sh)
	@echo "sim launching headless; give it ~10s, then check: make collect"

collect:                     ## scripted sim rollouts -> MP4 + demo .npz
	$(call RUN,python scripts/franka_sim_rollout_record.py --episodes 3 \
	  --out output_dir/franka/rollout.mp4 --data output_dir/franka/sim_demos.npz)

base-policy:                 ## BC-train the base policy from the collected sim demos
	$(call RUN,python scripts/build_base_policy.py --demos output_dir/franka/sim_demos.npz \
	  --save_path trained_models/franka/base_policy)

mile:                        ## iterative MILE run against the live sim
	$(call RUN,cd scripts && python train_mile.py --config ../config_franka.json)
```

Note: Makefile recipe lines must be **tab-indented** (not spaces).

- [ ] **Step 3: Verify the shell verb lands in a ready environment**

Run:
```bash
make build && make up
make shell <<'EOF'
ros2 pkg prefix franka_description && python3 -c "import mile_franka; print('ok')"
EOF
```
Expected: prints the `franka_description` install prefix and `ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/in_container_env.sh Makefile
git commit -m "Add in-container env helper + Makefile verbs (build/up/down/shell)"
```

---

## Task 5: `sim_up.sh` + `make sim-up`

**Files:**
- Create: `scripts/sim_up.sh`

The stacking scene loads `panda.xml` + `stacking_objects.xml` via **relative** includes, so both
XMLs must sit in `franka_description/mujoco/franka/` at runtime. We copy them there (does not
edit the source repo) and launch our wrapper with `scene_path` pointing at the copy.

- [ ] **Step 1: Write the launch script**

Create `scripts/sim_up.sh`:

```bash
#!/usr/bin/env bash
# Launch the multipanda MuJoCo stacking sim headless. Invoke via `make sim-up` (runs detached).
set -e
source /home/user/mile-code/scripts/in_container_env.sh

# 1. Inject the cube scene beside franka_description's panda.xml (relative includes need this).
FD=$(python3 -c "from ament_index_python.packages import get_package_share_directory as g; print(g('franka_description'))")
DEST="$FD/mujoco/franka"
cp /home/user/mile-code/mile_franka/assets/mujoco/stacking_scene.xml   "$DEST/"
cp /home/user/mile-code/mile_franka/assets/mujoco/stacking_objects.xml "$DEST/"
SCENE="$DEST/stacking_scene.xml"

# 2. Headless display (idempotent).
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
sleep 1

# 3. Launch our wrapper (cubes + inactive move-to-start / cartesian-impedance controllers).
exec ros2 launch /home/user/mile-code/mile_franka/launch/franka_sim_stacking.launch.py \
    scene_path:="$SCENE"
```

- [ ] **Step 2: Bring the sim up and verify the interfaces**

Run:
```bash
make sim-up
sleep 12
docker compose -f docker/docker-compose.yml exec sim bash -lc \
  'source scripts/in_container_env.sh && ros2 service list | grep -E "get_body_state|set_pause" && \
   ros2 topic list | grep equilibrium_pose'
```
Expected: `/get_body_state`, `/set_pause`, and `/cartesian_impedance/equilibrium_pose` all listed.

- [ ] **Step 3: Verify cube body poses are readable**

Run:
```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc \
  'source scripts/in_container_env.sh && \
   ros2 service call /get_body_state mujoco_ros_msgs/srv/GetBodyState "{name: top_cube}" | head -20'
```
Expected: a successful response with a non-trivial `pose` for `top_cube`.

- [ ] **Step 4: Commit**

```bash
git add scripts/sim_up.sh
git commit -m "Add sim_up.sh + make sim-up (inject cube scene, Xvfb, launch stacking sim)"
```

---

## Task 6: Point the MILE config at the sim env

**Files:**
- Modify: `config_franka.json`

- [ ] **Step 1: Switch the env to the sim backend**

In `config_franka.json`, change the `env_name` line:

```json
        "env_name": "Franka-Stack-Sim-v0",
```

(Leave `collector: "real"`, `intervener: "scripted"`, `rollout.auto_eval: false`,
`policy_path: "../trained_models/franka/base_policy"`, and `save.outdir: "../output_dir/franka"`
unchanged.)

- [ ] **Step 2: Verify the config still parses and resolves the sim env id**

Run:
```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc \
  'source scripts/in_container_env.sh && python3 -c "
import json; from mile_franka.envs.registration import SIM_ENV_ID
c=json.load(open(\"config_franka.json\"))
assert c[\"experiment\"][\"env_name\"]==SIM_ENV_ID, c[\"experiment\"][\"env_name\"]
assert c[\"experiment\"][\"rollout\"][\"auto_eval\"] is False
print(\"config sim env OK\")"'
```
Expected: `config sim env OK`.

- [ ] **Step 3: Commit**

```bash
git add config_franka.json
git commit -m "Point config_franka.json at Franka-Stack-Sim-v0 (single sim path)"
```

---

## Task 7: Collect sim demos — coherence gate

**Files:**
- (Conditional) Modify: `mile_franka/envs/ros_backend.py`

This is the spec's feasibility gate: scripted rollouts **through `FrankaEnv.step`** (10 Hz delta
actions) are not yet confirmed to descend/grasp (only the slow waypoint demo is). The demos must
be *coherent* (approach → descend → close → lift → over base → release), not necessarily
successful — the base policy is mediocre by design.

- [ ] **Step 1: Collect with the sim already up**

Run (requires `make sim-up` from Task 5 to be live):
```bash
make collect
```
Expected: prints `episode 0/1/2 …`, a final `success rate = …`, and
`saved … transitions -> output_dir/franka/sim_demos.npz` plus `… -> output_dir/franka/rollout.mp4`.

- [ ] **Step 2: Verify the saved demo array shape**

Run:
```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc \
  'source scripts/in_container_env.sh && python3 -c "
import numpy as np; d=np.load(\"output_dir/franka/sim_demos.npz\")
print(\"obs\", d[\"obs\"].shape, \"acts\", d[\"acts\"].shape, \"episodes\", int(d[\"episode\"].max())+1)
assert d[\"obs\"].shape[1]==72 and d[\"acts\"].shape[1]==4"'
```
Expected: `obs (N, 72) acts (N, 4) episodes 3`.

- [ ] **Step 3: Review the rollout video for coherence**

Open `output_dir/franka/rollout.mp4` on the host (the bind mount makes it appear immediately).
Confirm the arm visibly: hovers over a cube, descends near cube-center, closes the gripper,
lifts, moves over the other cube, and opens. It need not stack reliably.

- [ ] **Step 4 (conditional): Tune the backend if the rollout is incoherent**

If the arm does not descend to cube-center or never grasps, the likely cause is the
`panda_hand`-vs-TCP ~10 cm obs offset and/or per-step settle time (spec §5.1). In
`mile_franka/envs/ros_backend.py`, apply the smallest fix that makes descent reach cube-center:
either subtract the constant hand→TCP z-offset when reporting EE z in `get_ee_position`, or
increase the settle in `set_equilibrium_pose` (`CONTROL_PERIOD_S`) so each delta tracks. Re-run
Steps 1–3 until the rollout is coherent. Record what you changed in the bring-up note (Task 10).

- [ ] **Step 5: Commit (only if Step 4 changed code)**

```bash
git add mile_franka/envs/ros_backend.py
git commit -m "Tune FrankaEnv sim descent so scripted delta rollouts reach cube-center"
```

---

## Task 8: `--demos` flag on `build_base_policy.py` + train the base policy

**Files:**
- Modify: `scripts/build_base_policy.py`

- [ ] **Step 1: Add the npz loader and the `--demos` branch**

In `scripts/build_base_policy.py`:

Replace the import block top (after the existing imports) — add `Transitions`, `SIM_ENV_ID`:

```python
from imitation.data.types import Transitions

from mile_franka.envs.registration import FAKE_ENV_ID, SIM_ENV_ID, register_franka_envs
```

Add this loader function above `main()`:

```python
def load_npz_transitions(path):
    """Build imitation Transitions from a franka_sim_rollout_record .npz (obs/acts/episode).

    next_obs/dones are reconstructed from episode boundaries: within an episode next_obs is the
    following obs; the last step of each episode is terminal (next_obs = itself, done = True).
    """
    d = np.load(path, allow_pickle=False)
    obs = d["obs"].astype(np.float32)
    acts = d["acts"].astype(np.float32)
    ep = d["episode"]
    n = len(obs)
    next_obs = np.empty_like(obs)
    dones = np.zeros(n, dtype=bool)
    for i in range(n):
        if i + 1 < n and ep[i + 1] == ep[i]:
            next_obs[i] = obs[i + 1]
        else:
            next_obs[i] = obs[i]
            dones[i] = True
    return Transitions(obs=obs, acts=acts, infos=np.array([{}] * n),
                       next_obs=next_obs, dones=dones)
```

Change the default env to the sim and add the flag — edit the argparse block:

```python
    ap.add_argument("--env_id", default=SIM_ENV_ID)
    ap.add_argument("--demos", default=None,
                    help="path to a sim-demo .npz; skips scripted collection when set")
```

Replace the demo-collection lines in `main()`:

```python
    if args.demos:
        print(f"Loading demos from {args.demos} ...")
        demos = load_npz_transitions(args.demos)
    else:
        scripted = ScriptedStackPolicy(getattr(env.unwrapped, "config", None),
                                       mediocre=args.mediocre)
        print(f"Collecting {args.demo_episodes} scripted demos (mediocre={args.mediocre})...")
        demos = collect_scripted_demos(env, scripted, args.demo_episodes, rng)
    print(f"Using {len(demos.obs)} transitions; BC training {args.bc_epochs} epochs...")
    policy = train_bc(demos, env.observation_space, env.action_space, rng,
                      n_epochs=args.bc_epochs)
```

(`FAKE_ENV_ID` stays imported but unused at the default; that is intentional so an explicit
`--env_id Franka-Stack-Fake-v0` still works for an offline import check.)

- [ ] **Step 2: Train the base policy from the collected demos**

Run (sim must be up; uses the npz from Task 7). Skip the live-sim eval for speed first:
```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc \
  'source scripts/in_container_env.sh && python scripts/build_base_policy.py \
     --demos output_dir/franka/sim_demos.npz --save_path trained_models/franka/base_policy \
     --eval_episodes 0'
```
Expected: `Using <N> transitions; BC training …`, then `Saved base policy to trained_models/franka/base_policy; success rate = 0.00` (eval skipped → 0.00 over 0 is fine).

- [ ] **Step 3: Verify the saved policy round-trips as an ActorCriticPolicy**

Run:
```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc \
  'source scripts/in_container_env.sh && python3 -c "
from stable_baselines3.common.policies import ActorCriticPolicy
p = ActorCriticPolicy.load(\"trained_models/franka/base_policy\")
print(\"loaded\", type(p).__name__, \"obs\", p.observation_space.shape)"'
```
Expected: `loaded ActorCriticPolicy obs (72,)`.

- [ ] **Step 4: Confirm it is mediocre (a short live-sim eval)**

Run:
```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc \
  'source scripts/in_container_env.sh && python scripts/build_base_policy.py \
     --demos output_dir/franka/sim_demos.npz --save_path /tmp/base_policy_eval --eval_episodes 5'
```
Expected: prints `success rate = <low but ≥0>` — mediocre as intended (re-trains to a temp path so it does not clobber the saved policy).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_base_policy.py
git commit -m "build_base_policy: add --demos npz flag, default to sim env"
```

---

## Task 9: MILE-readiness smoke (iterative run in sim)

**Files:** none (uses `config_franka.json` + the saved base policy)

- [ ] **Step 1: Run one-plus iterative rounds against the live sim**

Run (sim up; base policy saved):
```bash
make mile
```
Expected: the run loads the base policy from `../trained_models/franka/base_policy`, runs
`num_rounds: 2` × `episodes_per_round: 3` of `Collector(mode='intervention')` with the
`ScriptedIntervener`, trains, and exits **without** any autonomous `generate_rollout`
(because `rollout.auto_eval: false`). Output appears under `output_dir/franka/`.

- [ ] **Step 2: Verify no autonomous rollout fired and output was written**

Run:
```bash
ls -la output_dir/franka/
docker compose -f docker/docker-compose.yml exec sim bash -lc \
  'source scripts/in_container_env.sh && grep -rn "generate_rollout" output_dir/franka/*.txt 2>/dev/null; \
   echo "exit-check done"'
```
Expected: a populated `output_dir/franka/` (saved model/log), and no evidence of an autonomous
policy rollout during collection.

- [ ] **Step 3: Commit any output-path config tweaks (only if needed)**

If `make mile` revealed a path/round-count tweak in `config_franka.json`, apply it and:
```bash
git add config_franka.json
git commit -m "config_franka.json: finalize sim iterative-run parameters"
```

---

## Task 10: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md`

- [ ] **Step 1: Update CLAUDE.md commands + the headless-gate wording**

In `CLAUDE.md`, under the Franka commands section, replace the headless/fake-env description
with the docker verbs and note the fake env is no longer a gate. Add:

```markdown
# --- Franka block-stacking (single sim path, all in the MILE-on-multipanda image) ---
make build && make up         # build the image, start the persistent `sim` service
make sim-up                   # launch the multipanda MuJoCo stacking sim headless
make collect                  # scripted FrankaEnv rollouts -> MP4 + sim_demos.npz
make base-policy              # BC-train the mediocre base policy from sim_demos.npz
make mile                     # iterative MILE run (config_franka.json, Franka-Stack-Sim-v0)
make shell                    # interactive in-container shell (env sourced)
```

And update the line that calls `config_franka.json` "the headless end-to-end gate" to say it now
targets `Franka-Stack-Sim-v0` and runs in-container against a live `make sim-up` (the fake env is
retained only for import-level checks, not as a gate).

- [ ] **Step 2: Record bring-up findings**

In `docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md`, append a dated section noting:
the MILE image build (deps that needed pinning from Task 1), whether the `FrankaEnv` delta
rollout needed the descent/settle tuning from Task 7 (and what changed), and that the iterative
MILE run completed in-container.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md
git commit -m "Docs: docker verbs, single sim path, MILE bring-up findings"
```

---

## Self-Review Notes (for the implementer)

- **Order matters:** Tasks 5 (`sim-up`) → 7 (`collect`) → 8 (`base-policy`) → 9 (`mile`) all need
  the live sim from Task 5 running in the `mile_sim` service.
- **Coherence gate is the real risk** (Task 7). If the scripted `FrankaEnv` rollout flails, do not
  proceed to Task 8 with garbage demos — fix the descent/settle first.
- **Fake env is dropped as a path**, not deleted: `build_base_policy.py` keeps `FAKE_ENV_ID`
  importable for offline checks, but nothing in this plan gates on it.
