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
