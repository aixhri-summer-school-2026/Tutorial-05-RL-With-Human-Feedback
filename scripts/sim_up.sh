#!/usr/bin/env bash
# Launch the multipanda MuJoCo stacking sim headless. Invoke via `make sim-up` (runs detached).
set -e
source /home/user/mile-code/scripts/in_container_env.sh

# Headless path renders to the internal Xvfb display only. Pin :99 even if an inherited
# (SSH-forwarded / host) DISPLAY is present -- mujoco_ros' GLFW init fails against a display
# it cannot reach, which cascades into "controller_manager unavailable" and a never-stepping sim.
export DISPLAY=:99

# 0. Tear down any prior sim cleanly. The launch wrapper and the mujoco_node have DIFFERENT
#    cmdlines, so kill BOTH -- killing only the wrapper orphans mujoco_node, which keeps running
#    and publishing a stale /clock while a new one starts, leaving duplicate mujoco_server nodes
#    and a non-deterministic controller graph (a prime source of the cold-boot controller race).
pkill -f 'franka_sim_stacking' 2>/dev/null || true
pkill -f 'mujoco_ros/lib/mujoco_ros/mujoco_node' 2>/dev/null || true
sleep 3
pkill -9 -f 'mujoco_ros/lib/mujoco_ros/mujoco_node' 2>/dev/null || true
sleep 1
# Drop the ros2 daemon's cached discovery so stale participants from the old sim don't linger.
ros2 daemon stop >/dev/null 2>&1 || true

# 1. Inject the cube scene beside franka_description's panda.xml (relative includes need this).
FD=$(python3 -c "from ament_index_python.packages import get_package_share_directory as g; print(g('franka_description'))")
DEST="$FD/mujoco/franka"
cp /home/user/mile-code/mile_franka/assets/mujoco/stacking_scene.xml   "$DEST/"
cp /home/user/mile-code/mile_franka/assets/mujoco/stacking_objects.xml "$DEST/"
SCENE="$DEST/stacking_scene.xml"

# 2. Headless display (idempotent). Fail loudly if Xvfb is missing or never comes up -- otherwise
#    mujoco_node dies later with an opaque "Failed to initialize GLFW" and the whole graph wedges.
command -v Xvfb >/dev/null || { echo "FATAL: Xvfb not installed in image -- rebuild it: make build" >&2; exit 1; }
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
for _ in $(seq 1 20); do [ -S /tmp/.X11-unix/X99 ] && break; sleep 0.5; done
[ -S /tmp/.X11-unix/X99 ] || { echo "FATAL: Xvfb :99 did not come up" >&2; tail -n 20 /tmp/xvfb.log 2>/dev/null >&2; exit 1; }

# 3. Launch our wrapper (cubes + inactive move-to-start / cartesian-impedance controllers).
exec ros2 launch /home/user/mile-code/mile_franka/launch/franka_sim_stacking.launch.py \
    scene_path:="$SCENE"
