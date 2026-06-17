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
