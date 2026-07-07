# MILE Tutorial — Pre-Session Setup (Homework)

This guide walks you through the one-time setup needed before the 2-hour summer-school tutorial. **Do this at home**, not at the venue. Setup takes 30–60 minutes on Linux/x86 and up to 90 minutes on Mac ARM.

**Goal:** your laptop is ready to run the tutorial when you walk in. All four parts (MetaWorld → Franka fake → Franka sim → Franka real) will work seamlessly from one docker image.

> **Mac (Apple Silicon) users:** both the `multipanda_ros2` base and the MILE image target `linux/amd64`. Your machine will build and run them under Rosetta 2 emulation. Everything works — training runs on CPU, build takes longer. Follow the Mac-specific callouts below.

---

## Prerequisites

### Linux (Ubuntu 22.04+) — recommended

- **GPU:** NVIDIA GPU (e.g., RTX 3060+) with driver ≥525
  - Test: `nvidia-smi`
  - No GPU? Pair up with someone or use a venue tower — CPU training is ~5× slower
- **Docker & Compose:** Docker 20.10+ and Compose plugin v2.5+
  - Test: `docker compose --version`
  - Not installed? [Install Docker](https://docs.docker.com/engine/install/ubuntu/) and [Compose](https://docs.docker.com/compose/install/linux/)
- **Docker GPU access:** `docker run --rm --gpus all ubuntu nvidia-smi` must work without `sudo`
  - If it fails: `sudo usermod -aG docker $USER && newgrp docker`
- **Disk space:** ≥50 GB free (`df -h /`)
- **Internet:** stable connection (~2 GB download)

### Mac (Apple Silicon — M1/M2/M3/M4)

- **Docker Desktop** 4.25+ with Rosetta emulation enabled:
  1. Open Docker Desktop → Settings → **Features in development**
  2. Enable **"Use Rosetta for x86/amd64 emulation on Apple Silicon"**
  3. Apply & Restart
- **Disk space:** ≥60 GB free (emulated layers are larger)
- **No GPU passthrough** — training runs on CPU, which is fine for the tutorial
- **Build time:** ~60–90 min total (vs ~25 min on Linux/x86 with GPU)
- Set the default platform once so every `docker` command targets amd64:
  ```bash
  export DOCKER_DEFAULT_PLATFORM=linux/amd64
  # Add to ~/.zshrc to persist across sessions
  echo 'export DOCKER_DEFAULT_PLATFORM=linux/amd64' >> ~/.zshrc
  ```

---

## Step 1: Verify prerequisites

**Linux:**
```bash
lsb_release -a
nvidia-smi | head -3
docker --version && docker compose --version
df -h /
docker run --rm --gpus all ubuntu nvidia-smi
```

**Mac ARM:**
```bash
# Confirm Rosetta emulation is working
docker run --rm --platform linux/amd64 ubuntu uname -m
# Expected output: x86_64
docker --version && docker compose --version
df -h /
```

**All commands should succeed.** If not, fix before moving on.

---

## Step 2: Clone the tutorial repository

```bash
git clone https://github.com/rayray2002/mile-franka-tutorial.git
cd mile-franka-tutorial
```

This is your working directory for all subsequent steps.

---

## Step 3: Build the hucebot base image (one-time)

The tutorial image is built on top of `hucebot:franka-humble`, which contains the Franka controller stack and ROS dependencies. The `multipanda_ros2` Dockerfile explicitly targets `linux/amd64` — Rosetta handles this transparently on Mac.

```bash
# Clone the hucebot repository (in a sibling directory)
cd ..
git clone https://github.com/hucebot/multipanda_ros2.git
cd multipanda_ros2
```

**Linux (∼10 min):**
```bash
docker compose build
```

**Mac ARM (∼40–60 min):**
```bash
docker buildx build --platform linux/amd64 -t hucebot:franka-humble .
```

**Expected output:** final line should say `Successfully tagged hucebot:franka-humble` (or similar).

```bash
# Return to the tutorial repo
cd ../mile-franka-tutorial
```

---

## Step 4: Build the MILE tutorial image

**Linux (∼15–25 min):**
```bash
make build
```

**Mac ARM (∼20–30 min, after the base image is done):**
```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 make build
```

**Expected output:** final line should say something like `Successfully tagged mile:franka-humble`. If you see build errors, [check troubleshooting](#troubleshooting).

You now have a `mile:franka-humble` image ready to run all four tutorial parts.

---

## Step 5: Start the persistent service

Start the docker-compose service (the container runs in the background).

```bash
make up
```

**Expected output:** 
```
Creating network "mile-franka-tutorial_default" with the default driver
Creating mile_sim ... done
```

The `mile_sim` container is now running and ready to accept commands.

---

## Step 6: Verify artifacts are present

The pretrained models (MetaWorld policies and the Franka base policy) are **included in the repository** under `trained_models/`. No download is needed.

Verify they're on disk:

```bash
ls trained_models/initial_policy trained_models/expert_policy \
   trained_models/gt_mental_model trained_models/warm_started_mental_model \
   trained_models/franka/base_policy
```

All five files should be listed. If any are missing, re-clone the repository.

---

## Step 7: Run the readiness check (in container)

Verify that all imports work and artifacts are in place. This check runs **inside the container** (the dependencies aren't installed on your host):

```bash
make tutorial-check
```

**Expected output:**
```
  import metaworld: OK
  import mile: OK
  import mile_franka: OK
  import torch: OK
  import gymnasium: OK
  artifact trained_models/initial_policy: OK
  artifact trained_models/expert_policy: OK
  artifact trained_models/gt_mental_model: OK
  artifact trained_models/warm_started_mental_model: OK
  artifact trained_models/franka/base_policy: OK
tutorial-check OK — you're ready.
```

If you see `FAIL` on any import or `MISSING` on any artifact, stop and debug (see [troubleshooting](#troubleshooting) below). **Do not proceed to the tutorial if this check fails.**

---

## Step 8: (Optional) Quick sanity check — run a shell

To verify the container environment is fully set up, open an interactive shell:

```bash
make shell
```

You should see a bash prompt with the environment already sourced. Try:

```bash
python3 -c "import metaworld; import mile; print('OK')"
```

Then exit:

```bash
exit
```

---

## Troubleshooting

### `nvidia-smi` not found or driver not installed

Install the NVIDIA driver (525+):

```bash
sudo apt-get update && sudo apt-get install -y nvidia-driver-525
```

Reboot and test:

```bash
nvidia-smi
```

### Docker permission denied

Your user is not in the `docker` group. Fix it:

```bash
sudo usermod -aG docker $USER
newgrp docker
# Test:
docker ps
```

No output means `docker ps` succeeded.

### Docker image build fails

Common issues:

1. **Out of disk space:** check `df -h /`. Need ≥50 GB free.
2. **Internet dropout:** resume by re-running `make build` (it caches layers).
3. **Old docker daemon:** update: `sudo apt-get install -y docker.io docker-compose-plugin`.

### `tutorial-check` shows `import metaworld: FAIL`

Likely a container issue. Try:

```bash
make down
make up
make tutorial-check
```

If it persists, check `docker logs mile_sim` for errors.

### `tutorial-check` shows missing artifacts

If `trained_models/` files are missing, re-clone the repository — the models are committed in-tree.

### Mac ARM: `exec format error` or `no matching manifest`

The image was not built for `linux/amd64`. Confirm `DOCKER_DEFAULT_PLATFORM=linux/amd64` is set and rebuild:
```bash
make down
DOCKER_DEFAULT_PLATFORM=linux/amd64 make build
make up
```

### Mac ARM: build hangs or is extremely slow

Rosetta emulation is CPU-bound. The hucebot base build compiles MuJoCo, libfranka, and a full ROS workspace — 60–90 min total is normal. Run the build overnight if needed. Once built, the image is cached and restarts are fast.

### Mac ARM: `sim-gui` window does not open

X11 forwarding is not available by default on macOS. Install [XQuartz](https://www.xquartz.org/) and run:
```bash
xhost +localhost
# Then in docker-compose.yml, set DISPLAY=host.docker.internal:0
```
Alternatively, skip `sim-gui` entirely — `make eval-base` and `make tutorial-collect-train` run headless.

---

## What's installed?

After setup, you have:

- **mile:franka-humble** docker image
  - Python 3.10 + PyTorch
  - MetaWorld v2 (peg-insert environment)
  - MILE codebase + Franka stack
  - MuJoCo 2.3.7 (native MetaWorld version)
  - ROS 2 Humble + multipanda sim
  - All tutorial configs and scripts
  
- **trained_models/** directory
  - MetaWorld peg-insert policies (Part 1)
  - Franka base policy (Part 3)

---

## Before you arrive

1. ✅ Check that `make tutorial-check` succeeds.
2. ✅ Confirm `make shell` works (optional, but good to know).
3. ✅ Bring your laptop, power cord, and ethernet cable if possible.
4. ✅ Verify `trained_models/` contains all 5 model files (Step 6).

**If anything is unclear or fails,** contact the instructor **before the session** — don't wait to debug at the venue.
