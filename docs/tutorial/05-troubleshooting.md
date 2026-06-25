# Tutorial Troubleshooting & FAQ

This guide covers common issues during the summer-school tutorial and their fixes. **Check this first** before asking for help — most issues have a one-line fix.

---

## Rendering / Display Issues

### ❌ `make sim-gui` shows a black window or won't display

**Problem:** MuJoCo can't render on your system. You see a black window, or the simulator launches but nothing appears.

**Fix:** Enable software rendering:
```bash
export LIBGL_ALWAYS_SOFTWARE=1
make sim-gui
```

**Why:** On headless systems or with weak GPU drivers, hardware OpenGL rendering fails. Software rendering (via llvmpipe) is slower but always works.

---

### ❌ `make sim-gui` fails with "Cannot connect to X server"

**Problem:** The simulator can't connect to your display. You see: `Error: Cannot connect to X server :1` or similar.

**Fix (Linux with X11):**
```bash
# On your *host* machine (not in container), run:
xhost +local:root
# Then retry:
make sim-gui
```

**Why:** Docker containers need permission to access your X11 display. `xhost +local:root` grants local root containers access to your display.

**Note:** If you're on **Wayland** instead of X11, you'll need to use XWayland. Check with `echo $DISPLAY` and `echo $WAYLAND_DISPLAY`. Contact the instructors if this fails.

---

### ❌ GPU not detected in container

**Problem:** `nvidia-smi` inside the container returns no GPU, or `make sim-gui` is very slow.

**Symptoms:**
- `docker run --rm --gpus all nvidia/cuda:12.0-runtime-ubuntu22.04 nvidia-smi` fails
- Training is extremely slow
- MuJoCo rendering is sluggish

**Fix:** Install the NVIDIA Container Runtime:
```bash
# On your host machine:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Then try again:
```bash
make sim-gui
```

**Why:** Docker needs the NVIDIA Container Runtime to pass GPU devices into containers. Without it, the `gpus: all` directive in `docker-compose.yml` is ignored.

---

## Artifact & Data Issues

### ❌ `trained_models/` files missing

**Problem:** `make tutorial-check` shows `MISSING` for some artifact files.

**Fix:** The pretrained models are committed in the repository. Re-clone or check your git state:
```bash
git status trained_models/
```

If files are missing, pull the latest:
```bash
git checkout trained_models/
```

**Why:** The MetaWorld and Franka models are stored in the repo (not downloaded separately). If they're missing, your local clone may be out of date or the files were accidentally deleted.

---

## Training & Data Collection

### ❌ `make tutorial-collect-train` fails immediately

**Problem:** Training starts but fails with a connection error or "env not responding":
```
ConnectionRefusedError: ... or mujoco environment not up
```

**Fix:** Start the simulator in another terminal first:
```bash
# Terminal 1:
make sim-up

# Terminal 2 (after 5 seconds):
make tutorial-collect-train
```

**Why:** The training script needs the MuJoCo simulator running. `make sim-up` launches it headless; `make tutorial-collect-train` connects to it. They must run concurrently.

**Tip:** Use `tmux` or split terminals so both run side-by-side without switching windows.

---

### ❌ `eval-base` or `eval-mile` crashes with `failed to reset bottom_cube`

**Problem:** After a previous run was interrupted (Ctrl-C, timeout, or crash), the sim objects are in inconsistent positions. You see:
```
RuntimeError: failed to reset bottom_cube near [...]; last pose was [...]
```

**Fix:** Restart the simulator:
```bash
make sim-up
# Wait ~10 seconds, then retry:
make eval-base
```

**Why:** When a prior run is aborted mid-episode, the MuJoCo sim objects (cubes) stay where they landed. On the next reset, the backend tries to move them back to start positions but the physics doesn't converge fast enough. A fresh `make sim-up` restores a clean state.

---

### ❌ MetaWorld import fails with `ModuleNotFoundError`

**Problem:** You see:
```
ModuleNotFoundError: No module named 'metaworld'
```
Or:
```
gymnasium.error.RegistrationError: Cannot find the environment ...
```

**Fix:** Check your MetaWorld installation:
```bash
pip show metaworld
pip show gymnasium
```

You should see `gymnasium 0.29.1` and a custom metaworld path (not PyPI). If MetaWorld is from PyPI (version 0.3.x+), uninstall and reinstall:

```bash
pip uninstall metaworld -y
pip install -r requirements.txt  # Installs the pinned MetaWorld commit
```

**Why:** PyPI's `metaworld` requires `gymnasium>=1.1`, which breaks our pinned stack (`gymnasium==0.29.1`). The repo's `requirements.txt` installs a compatible v2-era commit via git.

**Inside container?** Run:
```bash
make shell
pip install -r requirements.txt
```

---

## Teleop & Keyboard Control

### ❌ Keys rotate the MuJoCo camera instead of moving the robot

**Problem:** You press W, A, S, D, etc. and the MuJoCo 3D view pans/rotates instead of the robot moving. The arm doesn't respond.

**Why:** MuJoCo viewer (`make sim-gui`) has its own key bindings for camera control (w=wireframe, a=axis, q/e=camera rotation, etc.) that overlap with the teleop keys. Only the focused window receives keystrokes, so if the MuJoCo window is focused it captures your keypresses instead of the teleop code.

**Fix:** Click the small **"MILE keyboard teleop"** pygame window (not the MuJoCo viewer) to give it keyboard focus. The teleop window is a small black/grey window labeled "MILE keyboard teleop — keep focused". After clicking it, your keys will reach the robot.

**If you can't find the teleop window:** It should have appeared when you ran `make tutorial-teleop` (or `make tutorial-collect-train`). If you can't see it, check:
- Did you run `xhost +local:root` on the host before `make shell`? (Required once per login)
- Is the window minimized or behind other windows?

---

### ❌ `make tutorial-teleop` or `make tutorial-collect-train` freezes (no output)

**Problem:** After printing `python3 scripts/tutorial_teleop.py`, nothing appears for a long time before the windows show up.

**Why:** This is expected — startup takes **~90 seconds**, and it is all ROS bringup, not the teleop code. Connecting to the sim (`gym.make`) waits for `controller_manager` to become responsive (up to 60 s on a cold boot) and then applies the controller gains via several `ros2 param set` calls. Each runs as a daemonless CLI subprocess that pays ~5 s of DDS discovery, so a handful of gains adds ~30 s. The teleop/keyboard layer itself adds well under a second. Nothing is frozen — wait it out.

**Fix:**
1. Wait 20+ seconds after `make sim-up` before running teleop, then allow ~90 s for the connection to complete.
2. If it's been more than ~2 minutes and nothing appeared, restart the sim:
   ```bash
   # On the host (not inside make shell):
   make sim-up
   # Wait 20 seconds, then retry inside make shell:
   make tutorial-teleop
   ```

---

### ❌ Keyboard input is ignored during `make tutorial-collect-train`

**Problem:** You press W, A, S, D, Space, G, etc., but the robot doesn't move (and the MuJoCo camera isn't moving either — confirming the teleop window is focused but keys aren't reaching the robot).

**Fix:** Click the "MILE keyboard teleop" pygame window to focus it:
```bash
# The window should appear. Click somewhere in it, then try keys again.
```

**Why:** pygame only captures keyboard input if the window has focus. Clicking the window gives it input focus.

**Debug tip:** Press a movement key and watch the terminal. If you see `key: W` logs but the robot doesn't move, focus is the issue.

---

## Docker & Container Issues

### ❌ Cannot pull `hucebot:franka-humble` base image

**Problem:** Docker build fails with:
```
denied: requested access to the resource is denied
```

Or:
```
Error response from daemon: pull access denied for hucebot/multipanda_ros2
```

**Symptoms:** `docker compose build` fails early.

**Fix:** The base image isn't on Docker Hub—you must build it locally:

```bash
# Clone hucebot/multipanda_ros2 locally (in a sibling directory):
cd ..
git clone https://github.com/hucebot/multipanda_ros2
cd multipanda_ros2
docker compose build  # Takes 10–15 min

# Return to the tutorial repo and rebuild:
cd ../mile-franka-tutorial
docker compose build
```

**Why:** The tutorial uses a custom Franka controller image that isn't publicly hosted. You must build it once, then the tutorial image layers on top.

---

## Python & Import Errors

### ❌ `torch` not found when running `make tutorial-check` on the host

**Problem:** You run `make tutorial-check` on your laptop and see:
```
ModuleNotFoundError: No module named 'torch'
```

**Fix:** Run the check **inside** the container:
```bash
make shell
make tutorial-check
```

Then `exit` to leave the container.

**Why:** `torch` and the MILE package are only installed in the Docker image, not on your host machine. The Dockerfile provides the environment; your host doesn't have it.

**Summary:** Always use `make shell` for Python work, then run commands inside.

---

## General Debugging

### Stuck or slow processes?

If a `make` target seems to hang, check these:

1. **Is the simulator running?** If a script needs `make sim-up`, confirm it's running in another terminal.
2. **Is the keyboard window focused?** Click it if you're in teleop mode.
3. **Check logs:** Ctrl+C to stop the process and read the terminal output. Copy any errors and search this troubleshooting guide.
4. **Restart containers:**
   ```bash
   make down
   docker compose build
   make up
   ```

---

## Still stuck?

If you can't find your issue here:

1. **Check the terminal output** — look for the first error message, not just the last line.
2. **Search the docs** — `docs/tutorial/` has detailed guides for each part.
3. **Ask the instructors** — describe:
   - What you ran (the command)
   - What you expected
   - What you got (paste the error)
   - Your OS, GPU, and Docker version

---
