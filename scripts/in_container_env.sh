#!/usr/bin/env bash
# Common in-container environment for the MILE/multipanda sim. `source` me before any verb.
# `bash -lc` does NOT source ~/.bashrc reliably, so set the sim's runtime deps explicitly
# (see the franka-sim-verified-bringup note).
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/user/Libraries/libfranka/lib:/home/user/Libraries/mujoco/lib"
source /opt/ros/humble/setup.bash
source /home/user/humble_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# CycloneDDS iceoryx shared-memory transport (see config/cyclonedds.xml).
# iox-roudi must be running before any ROS node starts.
# docker-compose boots it, but docker exec sessions need a guard.
if ! pgrep -x iox-roudi > /dev/null 2>&1; then
    LD_LIBRARY_PATH=/opt/ros/humble/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
        /opt/ros/humble/bin/iox-roudi -c /home/user/mile-code/config/roudi_config.toml &
    sleep 1  # let RouDi init its shared-memory segments
fi
export CYCLONEDDS_URI=file:///home/user/mile-code/config/cyclonedds.xml
export DISPLAY="${DISPLAY:-:99}"
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
cd /home/user/mile-code
