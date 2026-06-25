# MILE Tutorial — Pre-Session Setup (Homework)

This guide walks you through the one-time setup needed before the 2-hour summer-school tutorial. **Do this at home**, not at the venue. Setup takes 30–60 minutes depending on your internet and hardware; the longest step is the docker image build (∼20 min on a typical laptop with GPU).

**Goal:** your laptop is ready to run the tutorial when you walk in. All four tiers (MetaWorld → Franka fake → Franka sim → Franka real) will work seamlessly from one docker image.

---

## Prerequisites

Check that your machine meets these requirements **before** starting:

- **OS:** Ubuntu 22.04 LTS or newer
- **GPU:** NVIDIA GPU (e.g., RTX 3060+) with driver ≥525
  - Test: `nvidia-smi` (should print driver version and GPU list)
  - No GPU? [Pair up](#for-weaker-laptops-pair-up) with someone or use a venue tower
- **Docker & Compose:**
  - Docker 20.10+ and Compose plugin (v2.5+)
  - Test: `docker compose --version` (should show `Docker Compose version 2.x.x` or newer)
  - Not installed? [Install Docker](https://docs.docker.com/engine/install/ubuntu/) and [Compose](https://docs.docker.com/compose/install/linux/)
- **Disk space:** at least 50–100 GB free
  - Check: `df -h /` (look at the "Available" column)
- **Internet:** stable connection (∼2 GB download for the base image + artifacts)
- **Ethernet recommended** for the venue real-robot session (WiFi can drop)
- **GPU access:** your user can run docker with GPU support
  - Test: `docker run --rm --gpus all ubuntu nvidia-smi` (should work without `sudo`)
  - If it fails, run `sudo usermod -aG docker $USER && newgrp docker` then log back in

---

## Step 1: Verify prerequisites

Before proceeding, confirm:

```bash
# OS version
lsb_release -a
# Driver version
nvidia-smi | head -3
# Docker + Compose
docker --version && docker compose --version
# Disk space
df -h /
# Your user can use docker + GPU
docker run --rm --gpus all ubuntu nvidia-smi
```

**All commands should succeed.** If not, stop and fix before moving on.

---

## Step 2: Clone the tutorial repository

```bash
git clone https://github.com/rayray2002/mile-franka-tutorial.git
cd mile-franka-tutorial
```

This is your working directory for all subsequent steps.

---

## Step 3: Build the hucebot base image (one-time, ∼10 min)

The tutorial image is built on top of `hucebot:franka-humble`, which contains the Franka controller stack and ROS dependencies. You need to clone and build this base image once.

```bash
# Clone the hucebot repository (in a sibling directory)
cd ..
git clone https://github.com/hucebot/multipanda_ros2.git
cd multipanda_ros2

# Build the base image (this will take ∼10 minutes)
docker compose build
```

**Expected output:** final line should say `Successfully tagged hucebot:franka-humble` (or similar).

```bash
# Return to the tutorial repo
cd ../mile-franka-tutorial
```

---

## Step 4: Build the MILE tutorial image (∼15–25 min)

Now build the tutorial image on top of the hucebot base. This includes MetaWorld, MILE, and all Franka dependencies in a single image.

```bash
make build
```

**Expected output:** final line should say something like `Successfully tagged mile:franka-humble`. If you see build errors, [check troubleshooting](#troubleshooting).

You now have a `mile:franka-humble` image ready to run all four tutorial tiers.

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
  - MetaWorld peg-insert policies (Tier 1)
  - Franka base policy (Tier 3)

---

## Before you arrive

1. ✅ Check that `make tutorial-check` succeeds.
2. ✅ Confirm `make shell` works (optional, but good to know).
3. ✅ Bring your laptop, power cord, and ethernet cable if possible.
4. ✅ Verify `trained_models/` contains all 5 model files (Step 6).

**If anything is unclear or fails,** contact the instructor **before the session** — don't wait to debug at the venue.
