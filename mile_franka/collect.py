"""Unified human-in-the-loop intervention collector (spec sec 5.6).

Collector runs the base policy and lets an Intervener override it; every timestep is
recorded into the MILE Box dict via InterventionDatasetBuilder. It returns the same
(dataset, mean_score, mean_success) tuple as collect_synthetic_data, so train_mile swaps
it in directly. Interveners: ScriptedIntervener (headless scripted "human") and
TeleopIntervener (SpaceMouse/Vive via the Phase 1 TeleopDevice interface).
"""
from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from mile_franka.data.dataset import InterventionDatasetBuilder
from mile_franka.policies.scripted import ScriptedStackPolicy
from mile_franka.teleop.base import TeleopDevice


class ScriptedIntervener:
    """A scripted 'human': intervenes when the base action drifts from the expert action."""

    def __init__(self, expert: Optional[ScriptedStackPolicy] = None, threshold: float = 0.5):
        self.expert = expert if expert is not None else ScriptedStackPolicy(mediocre=False)
        self.threshold = threshold

    def reset(self, rng: np.random.Generator) -> None:
        self.expert.reset(rng)

    def intervene(self, obs: np.ndarray, rollout_action: np.ndarray
                  ) -> Tuple[np.ndarray, bool, bool]:
        expert_action = self.expert.act(np.asarray(obs)[-18:])
        drift = float(np.linalg.norm(expert_action[:3] - np.asarray(rollout_action)[:3]))
        grip_disagree = (expert_action[3] > 0) != (np.asarray(rollout_action)[3] > 0)
        intervene = drift > self.threshold or grip_disagree
        return expert_action, intervene, False


class TeleopIntervener:
    """Adapts a TeleopDevice (SpaceMouse/Vive) to the Intervener interface."""

    def __init__(self, device: TeleopDevice):
        self.device = device

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def intervene(self, obs: np.ndarray, rollout_action: np.ndarray
                  ) -> Tuple[np.ndarray, bool, bool]:
        reading = self.device.read()
        return np.asarray(reading.action, dtype=np.float32), bool(reading.intervene), bool(reading.done)


class Collector:
    """Runs the base policy under intervention; emits the MILE Box dataset dict."""

    def __init__(self, env: gym.Env):
        self.env = env

    def collect_intervention(self, policy, intervener, n_episodes: int,
                             max_t: Optional[int] = None):
        """Return (dataset_dict, mean_score, mean_success_rate) -- collect_synthetic_data shape."""
        max_t = max_t if max_t is not None else self.env.unwrapped.config.max_steps
        builder = InterventionDatasetBuilder()
        policy.eval()
        scores, successes = [], []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            if hasattr(intervener, "reset"):
                intervener.reset(self.env.np_random)
            score, success = 0.0, 0
            for _ in range(max_t):
                rollout_action, _ = policy.predict(np.asarray(state), deterministic=True)
                human_action, intervene, done_sig = intervener.intervene(state, rollout_action)
                exec_action = human_action if intervene else rollout_action
                next_state, reward, terminated, truncated, info = self.env.step(exec_action)
                done = bool(terminated or truncated or done_sig)
                builder.add(state=np.asarray(state), rollout_action=np.asarray(rollout_action),
                            action=np.asarray(exec_action), intervention=intervene,
                            reward=reward, next_state=np.asarray(next_state), done=done)
                state = next_state
                score += reward
                if done or info["success"]:
                    success = int(info["success"])
                    break
            scores.append(score)
            successes.append(success)
        return builder.to_dict(), float(np.mean(scores)), float(np.mean(successes))
