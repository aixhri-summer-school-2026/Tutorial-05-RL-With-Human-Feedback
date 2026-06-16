import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import (
    BOTTOM_CUBE, TOP_CUBE, FakeRobotBackend, FakeWorld, WorldPoseSource)
from mile_franka.envs.franka_env import FrankaEnv


def move_to(env, target_xyz, gripper, max_iters=200):
    """Step toward target_xyz with the given gripper command until close or timed out."""
    info = {"success": 0}
    for _ in range(max_iters):
        ee = env.backend.get_ee_position()
        delta = np.asarray(target_xyz, dtype=np.float32) - ee
        if np.linalg.norm(delta) < 1e-3:
            break
        step = np.clip(delta / env.config.action_scale, -1.0, 1.0)
        action = np.array([step[0], step[1], step[2], gripper], dtype=np.float32)
        _, _, _, _, info = env.step(action)
    return info


def main():
    c = StackTaskConfig()
    world = FakeWorld(c)
    env = FrankaEnv(FakeRobotBackend(world), WorldPoseSource(world), c)
    env.reset(seed=1)

    top = env.pose_source.get_pose(TOP_CUBE).to_array()[:3]
    bottom = env.pose_source.get_pose(BOTTOM_CUBE).to_array()[:3]
    lift_z = 0.30
    stack_z = c.table_z + c.cube_size / 2.0 + c.cube_size  # one cube above the bottom

    move_to(env, top, gripper=-1.0)                                   # descend onto top cube
    env.step(np.array([0, 0, 0, 1.0], dtype=np.float32))             # close -> grasp
    move_to(env, [top[0], top[1], lift_z], gripper=1.0)               # lift
    move_to(env, [bottom[0], bottom[1], lift_z], gripper=1.0)         # carry over bottom
    move_to(env, [bottom[0], bottom[1], stack_z], gripper=1.0)        # descend to stack height
    _, _, term, _, info = env.step(np.array([0, 0, 0, -1.0], dtype=np.float32))  # release

    assert info["success"] == 1, f"expected success, got {info}"
    assert term is True, "success should terminate the episode"
    print("smoke_franka_env ok")


if __name__ == "__main__":
    main()
