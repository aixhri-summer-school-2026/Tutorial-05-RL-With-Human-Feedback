# Single MILE-on-multipanda image: hucebot base + the pinned MILE training stack.
# MetaWorld-free (the sim's MuJoCo is C++-only; no mujoco<3 Python binding needed).
FROM hucebot:franka-humble
USER root

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
# ROS Python packages live under /opt/ros/humble and the user workspace; source both
# setup scripts so they are on PYTHONPATH for this non-interactive RUN step.
RUN bash -c "source /opt/ros/humble/setup.bash \
 && source /home/user/humble_ws/install/setup.bash \
 && python3 -c \"import rclpy, torch, stable_baselines3, imitation, gymnasium, mile, mile_franka; \
from mujoco_ros_msgs.srv import GetBodyState; \
assert gymnasium.__version__=='0.29.1', gymnasium.__version__; print('deps OK')\""

CMD ["sleep", "infinity"]
