# Phase 2 multipanda bring-up checklist (spec sec 8 gate 1, real backend)

Run when hucebot's multipanda_ros2 controller docker is up (lab MuJoCo sim) and again on
hardware (France). Resolve every `CONFIRM@bringup:` marker:
`grep -rn CONFIRM@bringup mile_franka`.

- [ ] **Controller docker:** start the **hucebot multipanda_ros2 controller image** (the
      France lab's image — do not roll your own). Run our ROS2 client on the same DDS graph
      (shared `ROS_DOMAIN_ID` or `network_mode: host`); confirm `ros2 node list` sees the
      controller node from our side.
- [ ] multipanda sim launched **inside that image**; `ros2 topic list` / `ros2 action list`
      show the equilibrium-pose topic, the FrankaState topic, and the gripper action. Copy
      the exact names off the running hucebot container into the constants in
      `mile_franka/envs/ros_backend.py`.
- [ ] Cube GT poses available: confirm whether the sim publishes per-object PoseStamped
      topics (wire them into `MujocoGtPoseSource`) or whether mjData must be read directly
      (then add that variant). Update `mile_franka/pose/mujoco_gt.py`.
- [ ] Down-facing wrist: publish `DOWN_QUAT` as the equilibrium orientation and confirm
      the gripper points at the table; correct `DOWN_QUAT` in `mile_franka/config.py` if not.
- [ ] Construct `MultipandaRosBackend` + `MujocoGtPoseSource` + `FrankaEnv`; `reset()`
      then `step()` a few hand-chosen actions; confirm obs shape (18,), EE moves, gripper
      opens/closes.
- [ ] Hand-drive a full stack (or replay the `move_to` sequence from
      `scripts/smoke_franka_env.py` against the real backend); confirm `info['success']`
      toggles to 1 when cubes are stacked.
- [ ] Confirm workspace bounds / `action_scale` feel right for the controller (spec sec 10);
      tune `StackTaskConfig` if needed.
