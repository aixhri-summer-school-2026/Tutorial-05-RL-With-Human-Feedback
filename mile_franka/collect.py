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

    def __init__(self, expert: Optional[ScriptedStackPolicy] = None, threshold: float = 0.5,
                 env: Optional[gym.Env] = None):
        self.expert = expert if expert is not None else ScriptedStackPolicy(mediocre=False)
        self.threshold = threshold
        # The scripted expert plans over full GT poses; the policy observation is reduced and
        # no longer carries them. Read the env's privileged_frame when available; fall back to
        # the legacy 18-dim obs tail for callers that still pass a full frame.
        self.env = env

    def reset(self, rng: np.random.Generator) -> None:
        self.expert.reset(rng)

    def intervene(self, obs: np.ndarray, rollout_action: np.ndarray
                  ) -> Tuple[np.ndarray, bool, bool]:
        frame = (self.env.unwrapped.privileged_frame() if self.env is not None
                 else np.asarray(obs)[-18:])
        expert_action = self.expert.act(frame)
        drift = float(np.linalg.norm(expert_action[:3] - np.asarray(rollout_action)[:3]))
        grip_disagree = (expert_action[3] > 0) != (np.asarray(rollout_action)[3] > 0)
        intervene = drift > self.threshold or grip_disagree
        return expert_action, intervene, False, False


class TeleopIntervener:
    """Adapts a TeleopDevice (SpaceMouse/Vive) to the Intervener interface."""

    def __init__(self, device: TeleopDevice):
        self.device = device
        self._intervening: bool = False

    def reset(self, rng: np.random.Generator) -> None:
        self._intervening = False
        if hasattr(self.device, 'reset'):
            self.device.reset()

    def intervene(self, obs: np.ndarray, rollout_action: np.ndarray
                  ) -> Tuple[np.ndarray, bool, bool, bool]:
        reading = self.device.read()
        action = np.asarray(reading.action, dtype=np.float32).copy()

        # On segment entry, sync device gripper state from the policy's current command
        # so the gripper starts matching the robot rather than a stale toggle value.
        if reading.intervene and not self._intervening:
            if hasattr(self.device, 'sync_gripper_state'):
                self.device.sync_gripper_state(float(rollout_action[3]) > 0)
            action[3] = float(rollout_action[3])  # also fix this frame's action

        self._intervening = bool(reading.intervene)
        return action, self._intervening, bool(reading.done), bool(reading.discard)


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
        ep = 0
        while ep < n_episodes:
            state, _ = self.env.reset()
            if hasattr(intervener, "reset"):
                intervener.reset(self.env.np_random)
            score, success = 0.0, 0
            ep_steps, ep_interventions = 0, 0
            prev_intervene = False
            prev_gripper = None
            seg_steps = 0
            ep_buffer: list = []  # staged until episode commits or discards
            discard_sig = False
            print(f"  episode {ep}: waiting for RB (Back=discard, Start=done)...")
            for _ in range(max_t):
                # Sample, don't take the mean: a mean action that collapses to ~0 at
                # OOD/grasp states stalls the rollout (deterministic closed-loop deadlock),
                # which would waste a whole MILE data-collection round. See build_base_policy.
                rollout_action, _ = policy.predict(np.asarray(state), deterministic=False)
                human_action, intervene, done_sig, discard_sig = intervener.intervene(
                    state, rollout_action)
                exec_action = human_action if intervene else rollout_action
                next_state, reward, terminated, truncated, info = self.env.step(exec_action)
                done = bool(terminated or done_sig)

                # segment transition logging
                if intervene and not prev_intervene:
                    seg_steps = 0
                    print(f"    [t={ep_steps:4d}] SEGMENT ON  — recording", flush=True)
                if not intervene and prev_intervene:
                    gripper = "close" if exec_action[3] > 0 else "open"
                    print(f"    [t={ep_steps:4d}] SEGMENT OFF — {seg_steps} steps recorded, "
                          f"gripper={gripper}", flush=True)

                # MILE learns from BOTH non-intervention (nu=0) and intervention
                # (nu=1) steps, so every step is recorded by default. Within an
                # intervention segment, skip "idle" frames (no translation and no
                # gripper change) so the human can pause and think without injecting
                # do-nothing actions as if they were intended demonstrations.
                dx, dy, dz = exec_action[0], exec_action[1], exec_action[2]
                moving = abs(dx) > 1e-4 or abs(dy) > 1e-4 or abs(dz) > 1e-4
                grip_closed = exec_action[3] > 0
                gripper_changed = prev_gripper is None or grip_closed != prev_gripper
                prev_gripper = grip_closed

                record = (not intervene) or moving or gripper_changed
                if record:
                    ep_buffer.append(dict(
                        state=np.asarray(state), rollout_action=np.asarray(rollout_action),
                        action=np.asarray(exec_action), intervention=intervene,
                        reward=reward, next_state=np.asarray(next_state), done=done))

                if intervene:
                    ep_interventions += 1
                    if record:
                        seg_steps += 1
                        gripper = "close" if grip_closed else "open "
                        print(f"    [t={ep_steps:4d}] dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} "
                              f"gripper={gripper}", flush=True)

                state = next_state
                score += reward
                ep_steps += 1
                prev_intervene = intervene
                if discard_sig or done or info["success"] or truncated:
                    success = int(info["success"])
                    break

            if discard_sig:
                print(f"  episode {ep}: DISCARDED ({len(ep_buffer)} steps thrown away) — retrying")
                continue  # don't increment ep; retry with a fresh env reset

            for step in ep_buffer:
                builder.add(**step)
            status = "SUCCESS" if success else ("done" if done_sig else "truncated")
            print(f"  episode {ep}: {status}  total_steps={ep_steps}  "
                  f"recorded={len(ep_buffer)} (interventions={ep_interventions})")
            scores.append(score)
            successes.append(success)
            ep += 1
        print(f"  round total: {len(builder)} human steps recorded across {n_episodes} episodes")
        return builder.to_dict(), float(np.mean(scores)), float(np.mean(successes))
