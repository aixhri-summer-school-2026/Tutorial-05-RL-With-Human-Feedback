"""Tutorial free-play teleop — practice keyboard controls before Tier 3.

Run this (inside the container, after `make sim-up`) to learn the controls
without collecting training data. Every episode is discarded on exit.

Controls:
  W/S = +X/-X (forward/back)   A/D = +Y/-Y (left/right)   Q/E = +Z/-Z (up/down)
  SPACE = toggle intervention segment   G = toggle gripper
  ENTER = end episode   BACKSPACE = discard episode   Ctrl-C = quit

The small pygame window must stay focused for key input to register — click it.
"""
import sys

import numpy as np
import gymnasium as gym

from mile_franka.envs.franka_env import FrankaEnv
from mile_franka.envs.registration import register_franka_envs, make_franka_env
from mile_franka.teleop.keyboard import KeyboardDevice


def main():
    register_franka_envs()
    franka_env: FrankaEnv = gym.make("Franka-Stack-Sim-v0")  # type: ignore[assignment]
    env = make_franka_env(franka_env)
    device = KeyboardDevice(translation_scale=0.02)

    print("=" * 60)
    print("Free-play teleop — practice mode (data NOT saved)")
    print("Click the pygame window to focus it, then use:")
    print("  W/S=±X  A/D=±Y  Q/E=±Z  SPACE=clutch  G=gripper")
    print("  ENTER=end episode  BACKSPACE=discard  Ctrl-C=quit")
    print("=" * 60)

    config = getattr(getattr(env, "unwrapped", env), "config", None)
    max_steps = getattr(config, "max_steps", 1000)

    ep = 0
    while True:
        env.reset()
        ep += 1
        print(f"\nEpisode {ep} started. Robot reset to home position.")
        step = 0
        for _ in range(max_steps):
            reading = device.read()
            action = np.asarray(reading.action, dtype=np.float32).copy()
            if np.isnan(action[3]):
                action[3] = -1.0  # gripper open until first G press

            _, _, terminated, _, info = env.step(action)
            step += 1

            if reading.discard:
                print(f"  Episode {ep} discarded after {step} steps.")
                break
            if reading.done or terminated:
                success = info.get("success", False)
                print(f"  Episode {ep} ended after {step} steps. success={success}")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nFree-play teleop ended. No data was saved.")
        sys.exit(0)
