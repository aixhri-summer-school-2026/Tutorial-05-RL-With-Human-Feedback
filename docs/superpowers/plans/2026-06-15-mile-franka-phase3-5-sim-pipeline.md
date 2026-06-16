# MILE-on-Franka — Phases 3–5: Sim Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is **implementation-focused (no TDD)** per project preference — write the code, smoke-check, commit.

**Goal:** Complete the in-lab sim pipeline so MILE's iterative intervention loop runs end-to-end against `FrankaEnv` with a mediocre BC base policy and a human-in-the-loop collector — fully exercised headless with a scripted "human", ready to swap in the SpaceMouse/real backend later.

**Architecture:** Phase 3 builds a deliberately-mediocre `ScriptedStackPolicy` (state machine over GT cube poses) and BC-distills it into the differentiable Gaussian `ActorCriticPolicy` MILE needs. Phase 4 adds a unified `Collector` that runs the base policy and overlays interventions from either a `ScriptedIntervener` (headless) or a `TeleopIntervener` (SpaceMouse/Vive), emitting the MILE Box dict via Phase 1's `InterventionDatasetBuilder`. Phase 5 makes surgical edits to `scripts/train_mile.py` / `mile/algorithm.py` / `mile/computational_model.py` to swap the synthetic collector for the real one, register the Franka env + its `COST_LOOKUP` entry, guard the autonomous `generate_rollout`, and stop requiring MetaWorld on the Franka path.

**Tech Stack:** Python 3.10, numpy, `gymnasium==0.29.1`, `stable-baselines3==2.3.2` (`ActorCriticPolicy`), `imitation==1.0.0` (`BC`, `Transitions`), torch. No ROS/MuJoCo — everything here runs against the Phase 2 `FakeWorld`.

**Spec:** `docs/superpowers/specs/2026-06-15-mile-franka-stacking-design.md` (§5.5 scripted/BC, §5.6 collector, §5.7 train edits, §8 gates 2–6)

**Builds on:**
- Phase 1 — `mile_franka/data/dataset.py::InterventionDatasetBuilder`, `BOX_KEYS`; `mile_franka/teleop/{base,spacemouse}.py`.
- Phase 2 — `mile_franka/envs/registration.py::{register_franka_envs, make_franka_env, FAKE_ENV_ID}`, `mile_franka/config.py::StackTaskConfig`, `mile_franka/envs/fake_backend.py::{TOP_CUBE, BOTTOM_CUBE}`.

---

## Verified codebase facts this plan depends on

Confirmed by reading/running the current repo — do not re-derive:

- **Obs layout (current frame = last 18 of the 72-vector).** `FrameStack(4)+Flatten`
  orders oldest→newest, so the **current** `FrankaEnv` frame is `obs[-18:]` =
  `[ee_xyz(3), gripper_width(1), top_pose(7), bottom_pose(7)]`; within a cube pose,
  position is the first 3. Verified empirically.
- **`BC` API (imitation 1.0.0):** `BC(*, observation_space, action_space, rng, policy=<ActorCriticPolicy>, demonstrations=<Transitions>, batch_size=...)`, then `bc.train(n_epochs=...)`; the trained net is `bc.policy`. `Transitions` fields: `obs, acts, infos, next_obs, dones`.
- **`collect_synthetic_data` return contract** (`scripts/collect_synthetic_interventions.py:27`): `(dataset_dict, mean_score, mean_success_rate)`. The Collector must match this exactly so the Phase 5 swap is drop-in. In the loop, `dataset['action']` = **executed** action (rollout action on ν=0, human action on ν=1); `dataset['rollout_action']` = base-policy action every step.
- **Trainer autonomous rollouts** (`mile/algorithm.py`): `__init__` calls `generate_rollout` unconditionally at line 186; `train()` calls it again at lines 332/338 when `rollout.enabled`. `train()` also references `best_policy` at the end **only guarded by `on_best_rollout_success_rate`** — if rollouts are disabled but that flag is on, it raises `NameError`. Both must be guarded together.
- **MetaWorld imports** are top-level in `scripts/train_mile.py` (line 3) and inside `scripts/collect_synthetic_interventions.py` — importing either currently requires MetaWorld. The Franka path must not.
- **`env.step` must return `info['success']`** — `FrankaEnv` already does.

## File structure (Phases 3–5)

```
mile_franka/
  policies/
    __init__.py
    scripted.py            # ScriptedStackPolicy (state machine) + ScriptedPolicyConfig
    bc.py                  # build_actor_critic_policy + train_bc
    demos.py               # collect_scripted_demos -> imitation Transitions
  collect.py               # Collector + ScriptedIntervener + TeleopIntervener
scripts/
  build_base_policy.py     # scripted demos -> BC -> save mediocre base policy + eval
config_franka.json         # iterative sim run config (Franka path, scripted human)
mile/computational_model.py  # MODIFY: add Franka COST_LOOKUP entries
scripts/train_mile.py        # MODIFY: lazy MetaWorld, Franka env branch, collector swap
mile/algorithm.py            # MODIFY: guard autonomous generate_rollout behind auto_eval
```

Responsibilities: `policies/scripted.py` = "deterministic mediocre stacker over GT poses";
`policies/bc.py` = "MILE-compatible ActorCriticPolicy + BC training"; `policies/demos.py` =
"scripted rollouts → Transitions"; `collect.py` = "base policy + intervention overlay →
MILE dict"; the three `MODIFY`s = "make `train_mile` run the Franka path without MetaWorld
and without autonomous execution".

---

# Phase 3 — Scripted stacking policy + BC base policy (spec §5.5, gates 2–3)

## Task 3.1: policies package + ScriptedStackPolicy

**Files:** create `mile_franka/policies/__init__.py`, `mile_franka/policies/scripted.py`.

A stateful state machine that reads one 18-dim `FrankaEnv` frame and emits a 4-DoF delta
action. The `mediocre` knobs (xy aim bias + per-episode random offset + action noise +
release-height error) make it start but not reliably finish — the regime MILE needs.

- [ ] **Step 1: Create the package init**

```bash
touch mile_franka/policies/__init__.py
```

- [ ] **Step 2: Write `mile_franka/policies/scripted.py`**

```python
"""Scripted block-stacking policy over ground-truth cube poses (spec sec 5.5).

A state machine: approach -> descend -> grasp -> lift -> over-base -> place -> release.
Reads one FrankaEnv frame (18,) and returns a 4-DoF delta action in [-1, 1]. With
mediocre=True it injects an aim bias, a per-episode random xy offset, action noise, and a
release-height error so it starts the task but cannot reliably finish -- the regime MILE
learns from. The skilled variant (mediocre=False) is the expert the ScriptedIntervener uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mile_franka.config import StackTaskConfig

# Indices into one FrankaEnv frame (18,): see Phase 2 observation convention.
EE = slice(0, 3)
GRIPPER_W = 3
TOP_POS = slice(4, 7)
BOTTOM_POS = slice(11, 14)

# State-machine phases.
_APPROACH, _DESCEND, _GRASP, _LIFT, _OVER_BASE, _PLACE, _RELEASE, _DONE = range(8)


@dataclass
class ScriptedPolicyConfig:
    hover_height: float = 0.20        # m above the table for transit
    pos_tol: float = 0.012            # within this of a sub-target -> advance phase
    dwell_steps: int = 3              # steps to hold while grasping/releasing
    # Mediocre knobs (ignored when mediocre=False):
    aim_xy_bias: float = 0.015        # constant horizontal placement bias (m)
    aim_xy_noise_std: float = 0.020   # per-episode random horizontal offset std (m)
    action_noise_std: float = 0.05    # gaussian noise on the [-1,1] action
    release_height_error: float = 0.015  # release this much too high (m)


class ScriptedStackPolicy:
    """Deterministic (optionally mediocre) cube stacker. Stateful: call reset() per episode."""

    def __init__(self, task: Optional[StackTaskConfig] = None,
                 cfg: Optional[ScriptedPolicyConfig] = None, mediocre: bool = True):
        self.task = task if task is not None else StackTaskConfig()
        self.cfg = cfg if cfg is not None else ScriptedPolicyConfig()
        self.mediocre = mediocre
        self._phase = _APPROACH
        self._dwell = 0
        self._episode_offset = np.zeros(2, dtype=np.float32)
        self._rng = np.random.default_rng()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self._phase = _APPROACH
        self._dwell = 0
        if rng is not None:
            self._rng = rng
        if self.mediocre:
            self._episode_offset = (
                np.array([self.cfg.aim_xy_bias, self.cfg.aim_xy_bias], dtype=np.float32)
                + self._rng.normal(0.0, self.cfg.aim_xy_noise_std, size=2).astype(np.float32))
        else:
            self._episode_offset = np.zeros(2, dtype=np.float32)

    def _delta(self, ee: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.clip((target - ee) / self.task.action_scale, -1.0, 1.0)

    def _reached(self, ee: np.ndarray, target: np.ndarray) -> bool:
        return bool(np.linalg.norm(target - ee) < self.cfg.pos_tol)

    def act(self, frame: np.ndarray) -> np.ndarray:
        """frame: a single FrankaEnv observation (18,). Returns action (4,) float32."""
        ee = np.asarray(frame[EE], dtype=np.float32)
        top = np.asarray(frame[TOP_POS], dtype=np.float32)
        bottom = np.asarray(frame[BOTTOM_POS], dtype=np.float32)
        hover_z = self.task.table_z + self.cfg.hover_height
        place_xy = bottom[:2] + (self._episode_offset if self.mediocre else 0.0)
        place_z = bottom[2] + self.task.cube_size + (
            self.cfg.release_height_error if self.mediocre else 0.0)

        gripper = -1.0  # default open
        if self._phase == _APPROACH:
            target = np.array([top[0], top[1], hover_z], dtype=np.float32)
            if self._reached(ee, target):
                self._phase = _DESCEND
        elif self._phase == _DESCEND:
            target = top.copy()
            if self._reached(ee, target):
                self._phase = _GRASP
        elif self._phase == _GRASP:
            target, gripper = ee.copy(), 1.0
            self._dwell += 1
            if self._dwell >= self.cfg.dwell_steps:
                self._dwell, self._phase = 0, _LIFT
        elif self._phase == _LIFT:
            target, gripper = np.array([ee[0], ee[1], hover_z], dtype=np.float32), 1.0
            if self._reached(ee, target):
                self._phase = _OVER_BASE
        elif self._phase == _OVER_BASE:
            target, gripper = np.array([place_xy[0], place_xy[1], hover_z], dtype=np.float32), 1.0
            if self._reached(ee, target):
                self._phase = _PLACE
        elif self._phase == _PLACE:
            target, gripper = np.array([place_xy[0], place_xy[1], place_z], dtype=np.float32), 1.0
            if self._reached(ee, target):
                self._phase = _RELEASE
        elif self._phase == _RELEASE:
            target, gripper = ee.copy(), -1.0
            self._dwell += 1
            if self._dwell >= self.cfg.dwell_steps:
                self._dwell, self._phase = 0, _DONE
        else:  # _DONE
            target, gripper = ee.copy(), -1.0

        delta = self._delta(ee, target)
        if self.mediocre and self.cfg.action_noise_std > 0:
            delta = np.clip(
                delta + self._rng.normal(0.0, self.cfg.action_noise_std, size=3), -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], gripper], dtype=np.float32)
```

- [ ] **Step 3: Smoke-check the skilled policy stacks on the fake env**

Run:
```bash
/home/ray/miniforge3/envs/mile/bin/python -c "
import numpy as np
from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import FakeWorld, FakeRobotBackend, WorldPoseSource
from mile_franka.envs.franka_env import FrankaEnv
from mile_franka.policies.scripted import ScriptedStackPolicy
c = StackTaskConfig(); w = FakeWorld(c)
env = FrankaEnv(FakeRobotBackend(w), WorldPoseSource(w), c)
pol = ScriptedStackPolicy(c, mediocre=False)
obs, _ = env.reset(seed=2); pol.reset(env.np_random)
success = 0
for _ in range(c.max_steps):
    obs, r, term, trunc, info = env.step(pol.act(obs))
    if info['success']: success = 1
    if term or trunc: break
assert success == 1, 'skilled scripted policy should stack'
print('scripted skilled ok')
"
```
Expected: prints `scripted skilled ok`

- [ ] **Step 4: Commit**

```bash
git add mile_franka/policies/__init__.py mile_franka/policies/scripted.py
git commit -m "Add ScriptedStackPolicy state machine for block stacking"
```

---

## Task 3.2: BC policy builder + trainer

**Files:** create `mile_franka/policies/bc.py`.

Builds the exact `ActorCriticPolicy` arch MILE uses (net `[256,256]`,
`NormalizeFeaturesExtractor`+`RunningNorm`) so the saved base policy loads via
`ActorCriticPolicy.load`, and trains it with imitation `BC`.

- [ ] **Step 1: Write `mile_franka/policies/bc.py`**

```python
"""Build and BC-train the MILE-compatible base policy (spec sec 5.5).

build_actor_critic_policy reproduces the arch train_mile.py uses for bc policies/mental
models so the saved policy round-trips through ActorCriticPolicy.load. train_bc distills
scripted demos into it with imitation's BC, yielding the differentiable Gaussian rollout
policy MILE requires.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.util.networks import RunningNorm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn


def build_actor_critic_policy(observation_space: gym.spaces.Space,
                              action_space: gym.spaces.Space) -> ActorCriticPolicy:
    """ActorCriticPolicy matching train_mile.py's bc arch (net [256,256] + RunningNorm)."""
    return ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=get_schedule_fn(1),
        net_arch=[256, 256],
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    )


def train_bc(transitions: Transitions, observation_space: gym.spaces.Space,
             action_space: gym.spaces.Space, rng: np.random.Generator,
             n_epochs: int = 20, batch_size: int = 64) -> ActorCriticPolicy:
    """Distill demonstrations into a fresh MILE-compatible ActorCriticPolicy."""
    policy = build_actor_critic_policy(observation_space, action_space)
    bc = BC(
        observation_space=observation_space,
        action_space=action_space,
        rng=rng,
        policy=policy,
        demonstrations=transitions,
        batch_size=batch_size,
    )
    bc.train(n_epochs=n_epochs, progress_bar=False)
    return bc.policy
```

- [ ] **Step 2: Smoke-check the policy builds and is loadable after save**

Run:
```bash
/home/ray/miniforge3/envs/mile/bin/python -c "
import tempfile, os, numpy as np, gymnasium as gym
from mile_franka.policies.bc import build_actor_critic_policy
from stable_baselines3.common.policies import ActorCriticPolicy
obs_space = gym.spaces.Box(-np.inf, np.inf, (72,), np.float32)
act_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
p = build_actor_critic_policy(obs_space, act_space)
d = tempfile.mkdtemp(); path = os.path.join(d, 'p')
p.save(path)
p2 = ActorCriticPolicy.load(path)
a, _ = p2.predict(np.zeros(72, dtype=np.float32), deterministic=True)
assert a.shape == (4,)
print('bc builder ok')
"
```
Expected: prints `bc builder ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/policies/bc.py
git commit -m "Add MILE-compatible ActorCriticPolicy builder and BC trainer"
```

---

## Task 3.3: Scripted demo collection

**Files:** create `mile_franka/policies/demos.py`.

Rolls the scripted policy on the **wrapped** (72-dim) env and packages `(obs, act)` into
imitation `Transitions`. The scripted policy reads `obs[-18:]` (the current frame).

- [ ] **Step 1: Write `mile_franka/policies/demos.py`**

```python
"""Collect scripted-policy rollouts as imitation Transitions (spec sec 5.5).

Runs ScriptedStackPolicy on the wrapped (72,) FrankaEnv and records every (obs, action,
next_obs, done) step. The policy reads the current frame obs[-18:].
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from imitation.data.types import Transitions

from mile_franka.policies.scripted import ScriptedStackPolicy


def collect_scripted_demos(env: gym.Env, policy: ScriptedStackPolicy,
                           n_episodes: int, rng: np.random.Generator) -> Transitions:
    """Roll `policy` on wrapped `env` for n_episodes; return imitation Transitions."""
    obs_list, act_list, next_list, done_list = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        policy.reset(rng)
        for _ in range(env.unwrapped.config.max_steps):
            action = policy.act(np.asarray(obs)[-18:])
            next_obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(np.asarray(action, dtype=np.float32))
            next_list.append(np.asarray(next_obs, dtype=np.float32))
            done_list.append(done)
            obs = next_obs
            if done or info["success"]:
                break
    n = len(obs_list)
    return Transitions(
        obs=np.array(obs_list, dtype=np.float32),
        acts=np.array(act_list, dtype=np.float32),
        infos=np.array([{}] * n),
        next_obs=np.array(next_list, dtype=np.float32),
        dones=np.array(done_list, dtype=bool),
    )
```

- [ ] **Step 2: Smoke-check demos collect with the right shapes**

Run:
```bash
/home/ray/miniforge3/envs/mile/bin/python -c "
import numpy as np, gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack
from mile_franka.envs.registration import register_franka_envs, FAKE_ENV_ID
from mile_franka.policies.scripted import ScriptedStackPolicy
from mile_franka.policies.demos import collect_scripted_demos
register_franka_envs()
env = FlattenObservation(FrameStack(gym.make(FAKE_ENV_ID), 4))
t = collect_scripted_demos(env, ScriptedStackPolicy(mediocre=False), 3, np.random.default_rng(0))
assert t.obs.shape[1] == 72 and t.acts.shape[1] == 4 and len(t.obs) == len(t.acts)
print('demos ok', len(t.obs))
"
```
Expected: prints `demos ok <N>` (N > 0)

- [ ] **Step 3: Commit**

```bash
git add mile_franka/policies/demos.py
git commit -m "Add scripted demo collection returning imitation Transitions"
```

---

## Task 3.4: build_base_policy script (spec §8 gate 3)

**Files:** create `scripts/build_base_policy.py`.

End-to-end: collect mediocre scripted demos → BC-train → save → eval the base policy's
success rate (should be mediocre, i.e. `0 < rate < 1`).

- [ ] **Step 1: Write `scripts/build_base_policy.py`**

```python
"""Build the mediocre BC base policy for MILE-on-Franka (spec sec 5.5, gate 3).

Collects scripted (mediocre) demos on the fake FrankaEnv, BC-distills an ActorCriticPolicy,
saves it, and reports its rollout success rate.
"""
import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack

from mile_franka.envs.registration import FAKE_ENV_ID, register_franka_envs
from mile_franka.policies.bc import train_bc
from mile_franka.policies.demos import collect_scripted_demos
from mile_franka.policies.scripted import ScriptedStackPolicy


def eval_policy(env, policy, n_episodes, rng):
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(env.unwrapped.config.max_steps):
            action, _ = policy.predict(np.asarray(obs), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if info["success"]:
                successes += 1
                break
            if terminated or truncated:
                break
    return successes / n_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default=FAKE_ENV_ID)
    ap.add_argument("--demo_episodes", type=int, default=60)
    ap.add_argument("--bc_epochs", type=int, default=30)
    ap.add_argument("--eval_episodes", type=int, default=30)
    ap.add_argument("--mediocre", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--save_path", default="trained_models/franka/base_policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    register_franka_envs()
    env = FlattenObservation(FrameStack(gym.make(args.env_id), 4))
    rng = np.random.default_rng(args.seed)

    scripted = ScriptedStackPolicy(mediocre=args.mediocre)
    print(f"Collecting {args.demo_episodes} scripted demos (mediocre={args.mediocre})...")
    demos = collect_scripted_demos(env, scripted, args.demo_episodes, rng)
    print(f"Collected {len(demos.obs)} transitions; BC training {args.bc_epochs} epochs...")
    policy = train_bc(demos, env.observation_space, env.action_space, rng,
                      n_epochs=args.bc_epochs)

    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    policy.save(args.save_path)
    rate = eval_policy(env, policy, args.eval_episodes, rng)
    print(f"Saved base policy to {args.save_path}; success rate = {rate:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and confirm a mediocre base policy is produced**

Run: `/home/ray/miniforge3/envs/mile/bin/python scripts/build_base_policy.py`
Expected: prints `Saved base policy to trained_models/franka/base_policy; success rate = <r>`
with `0.0 <= r < 1.0` (mediocre — it should not solve every episode). If `r` is `0.00`,
lower `ScriptedPolicyConfig.aim_xy_noise_std`/`release_height_error`; if `1.00`, raise them.

- [ ] **Step 3: Confirm the saved policy reloads as an ActorCriticPolicy**

Run:
```bash
/home/ray/miniforge3/envs/mile/bin/python -c "
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
p = ActorCriticPolicy.load('trained_models/franka/base_policy')
a, _ = p.predict(np.zeros(72, dtype=np.float32), deterministic=True)
assert a.shape == (4,)
print('base policy reload ok')
"
```
Expected: prints `base policy reload ok`

- [ ] **Step 4: Commit (script only — the model artifact is git-ignored output)**

```bash
git add scripts/build_base_policy.py
git commit -m "Add build_base_policy: scripted demos -> BC -> mediocre base policy"
```

---

# Phase 4 — Human-in-the-loop Collector (spec §5.6, gate 4)

## Task 4.1: Collector + interveners

**Files:** create `mile_franka/collect.py`.

Replaces `collect_synthetic_data`. The base policy drives; an **intervener** can take over.
`ScriptedIntervener` (headless: a skilled scripted "human" that grabs control when the
base action drifts) and `TeleopIntervener` (adapts any `TeleopDevice`) share one interface,
so the collector is identical headless and with a SpaceMouse/Vive.

- [ ] **Step 1: Write `mile_franka/collect.py`**

```python
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
```

- [ ] **Step 2: Smoke-check the headless collector matches the MILE schema**

Uses a fresh (untrained) base policy so it drifts and triggers interventions — no Phase 3
artifact needed.

Run:
```bash
/home/ray/miniforge3/envs/mile/bin/python -c "
import numpy as np, gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack
from mile_franka.envs.registration import register_franka_envs, FAKE_ENV_ID
from mile_franka.policies.bc import build_actor_critic_policy
from mile_franka.collect import Collector, ScriptedIntervener
from mile_franka.data.dataset import BOX_KEYS
from mile.utils import prepare_dataset, DictDataset
register_franka_envs()
env = FlattenObservation(FrameStack(gym.make(FAKE_ENV_ID), 4))
policy = build_actor_critic_policy(env.observation_space, env.action_space)
data, score, succ = Collector(env).collect_intervention(policy, ScriptedIntervener(), n_episodes=3)
assert set(data.keys()) == set(BOX_KEYS), data.keys()
ivals = set(data['intervention'])
assert 0 in ivals and 1 in ivals, f'need both nu values, got {ivals}'
train, valid = prepare_dataset(data)
assert len(DictDataset(train)) > 0
print('collector ok', len(data['state']), 'score', round(score,2), 'succ', succ)
"
```
Expected: prints `collector ok <N> score <s> succ <r>` and does not assert.

- [ ] **Step 3: Commit**

```bash
git add mile_franka/collect.py
git commit -m "Add human-in-the-loop Collector with scripted/teleop interveners"
```

---

# Phase 5 — train_mile integration (spec §5.7, gates 5–6)

## Task 5.1: Add Franka COST_LOOKUP entries

**Files:** modify `mile/computational_model.py:14-20`.

- [ ] **Step 1: Add the entries**

In `mile/computational_model.py`, extend `COST_LOOKUP` (start from `pick-place` per spec §4):

```python
COST_LOOKUP = {    
    'button-press-v2': [150, 200.0],
    'peg-insert-side-v2': [75, 175.0],
    'pick-place-v2': [250, 200.0],
    'drawer-open-v2': [60, 75.0],
    'LunarLander-v2': [3, 1.0],
    'Franka-Stack-Fake-v0': [250, 200.0],
    'Franka-Stack-Sim-v0': [250, 200.0],
}
```

- [ ] **Step 2: Smoke-check the entries are present**

Run:
```bash
/home/ray/miniforge3/envs/mile/bin/python -c "
from mile.computational_model import COST_LOOKUP
assert COST_LOOKUP['Franka-Stack-Fake-v0'] == [250, 200.0]
print('cost lookup ok')
"
```
Expected: prints `cost lookup ok`

- [ ] **Step 3: Commit**

```bash
git add mile/computational_model.py
git commit -m "Add Franka stacking entries to COST_LOOKUP"
```

---

## Task 5.2: Guard autonomous generate_rollout behind auto_eval

**Files:** modify `mile/algorithm.py` (`InterventionTrainer.__init__` ~line 185, `train()` ~lines 320-365).

On a real robot the auto-rollout autonomously executes the policy (unsafe/slow). Gate every
`generate_rollout` behind `experiment.rollout.auto_eval` (default `True` to preserve the
MetaWorld behavior), and fix the `best_policy` `NameError` when rollouts are off.

- [ ] **Step 1: Guard the `__init__` rollout**

Replace these lines in `__init__` (currently lines ~185-187):

```python
        self.score_window = deque(maxlen=100)
        self.init_policy = deepcopy(self.policy)
        self.score_window, init_success_rate = generate_rollout(self.policy, self.env, env_name=self.env_name, scores_window=self.score_window)
        self.logger.log_rollout(success_rate=init_success_rate, init_success_rate=init_success_rate)
```

with:

```python
        self.score_window = deque(maxlen=100)
        self.init_policy = deepcopy(self.policy)
        self.auto_eval = config['experiment'].get('rollout', {}).get('auto_eval', True)
        if self.auto_eval:
            self.score_window, init_success_rate = generate_rollout(self.policy, self.env, env_name=self.env_name, scores_window=self.score_window)
            self.logger.log_rollout(success_rate=init_success_rate, init_success_rate=init_success_rate)
```

- [ ] **Step 2: Guard the in-loop rollout and the best_policy save**

In `train()`, initialize the best-trackers before the epoch loop. Change:

```python
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, round: Optional[int]=None):
        best_success_rate = 0
```

to:

```python
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, round: Optional[int]=None):
        best_success_rate = 0
        best_policy = None
        best_mental_model = None
```

Then gate the rollout block — change:

```python
            if self.experiment_config['rollout']['enabled']:
```

to:

```python
            if self.experiment_config['rollout']['enabled'] and self.auto_eval:
```

Finally, guard the best-policy save at the end of `train()` — change:

```python
            if self.experiment_config['save']['on_best_rollout_success_rate']:
                best_policy.save(self.experiment_config['save']['outdir']+'/best_policy')
                best_mental_model.save(self.experiment_config['save']['outdir']+'/best_mental_model')
```

to:

```python
            if self.experiment_config['save']['on_best_rollout_success_rate'] and best_policy is not None:
                best_policy.save(self.experiment_config['save']['outdir']+'/best_policy')
                best_mental_model.save(self.experiment_config['save']['outdir']+'/best_mental_model')
```

- [ ] **Step 3: Smoke-check the module still imports**

Run: `/home/ray/miniforge3/envs/mile/bin/python -c "import mile.algorithm; print('algorithm ok')"`
Expected: prints `algorithm ok`

- [ ] **Step 4: Commit**

```bash
git add mile/algorithm.py
git commit -m "Guard autonomous generate_rollout behind rollout.auto_eval flag"
```

---

## Task 5.3: Make MetaWorld optional and add the Franka env branch + collector swap

**Files:** modify `scripts/train_mile.py`.

Three surgical edits so the Franka path runs without MetaWorld and with the real collector.

- [ ] **Step 1: Make the MetaWorld import lazy**

Change the top-level import (line 3):

```python
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN # type: ignore
```

to:

```python
try:
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN # type: ignore
except Exception:  # MetaWorld is not installed on the Franka path
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN = {}, {}
```

And delete the top-level `from collect_synthetic_interventions import collect_synthetic_data`
(it pulls MetaWorld); it is re-imported lazily in Step 4.

- [ ] **Step 2: Add a shared env builder**

Add this helper near the top of `scripts/train_mile.py` (after the imports):

```python
def build_franka_or_metaworld_env(env_name):
    """Build the wrapped (FrameStack+Flatten) training env for either backend."""
    if env_name + '-goal-observable' in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']()
        env._freeze_rand_vec = False
        env = FrameStack(env, 4)
        env = FlattenObservation(env)
    elif env_name.startswith('Franka'):
        from mile_franka.envs.registration import register_franka_envs, make_franka_env
        register_franka_envs()
        env = make_franka_env(gym.make(env_name))
    else:
        env = gym.make(env_name)
    return Monitor(env)
```

Then in `iterative_training`, replace the env-construction block:

```python
    env_name = config['experiment']['env_name']
    if env_name+'-goal-observable' in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+'-goal-observable']()
        env._freeze_rand_vec = False
        env = FrameStack(env, 4)
        env = FlattenObservation(env)
    else:
        env = gym.make(env_name)
    env = Monitor(env)
    env.reset()
```

with:

```python
    env_name = config['experiment']['env_name']
    env = build_franka_or_metaworld_env(env_name)
    env.reset()
```

- [ ] **Step 3: Gate the synthetic-only model loads**

In `iterative_training`, the `intervention_policy` and `gt_mental_model` are only needed by
the synthetic collector. Wrap their loads. Change the `policy_type == 'bc'` block:

```python
    elif config['experiment']['policy_type'] == 'bc':
        policy = ActorCriticPolicy.load(config['experiment']['policy_path'])
        intervention_policy = ActorCriticPolicy.load(config['experiment']['intervention_policy_path'])
        intervention_policy.eval()
```

to:

```python
    elif config['experiment']['policy_type'] == 'bc':
        policy = ActorCriticPolicy.load(config['experiment']['policy_path'])
        intervention_policy = None
        if config['experiment'].get('collector', 'synthetic') == 'synthetic':
            intervention_policy = ActorCriticPolicy.load(config['experiment']['intervention_policy_path'])
            intervention_policy.eval()
```

(Apply the same `intervention_policy = None` default in the `sac` and `qnetwork` branches if
you run those on the Franka path; for the cube tutorial only `bc` is used.)

Likewise gate `gt_mental_model`. Change the mental-model `bc` block:

```python
        gt_mental_model = ActorCriticPolicy.load(config['experiment']['gt_mental_model_path'])
        gt_mental_model.eval()
```

to:

```python
        gt_mental_model = None
        if config['experiment'].get('collector', 'synthetic') == 'synthetic':
            gt_mental_model = ActorCriticPolicy.load(config['experiment']['gt_mental_model_path'])
            gt_mental_model.eval()
```

And guard the unconditional `gt_mental_model.to(device)` / `intervention_policy.to(device)`
calls (they follow the loads):

```python
    policy.to(device)
    intervention_policy.to(device)
    mental_model.to(device)
    gt_mental_model.to(device)
```

to:

```python
    policy.to(device)
    if intervention_policy is not None:
        intervention_policy.to(device)
    mental_model.to(device)
    if gt_mental_model is not None:
        gt_mental_model.to(device)
```

- [ ] **Step 4: Swap the collector inside the round loop**

Before the `for round in range(num_rounds):` loop, build the real collector + intervener:

```python
    collector_type = config['experiment'].get('collector', 'synthetic')
    real_collector = None
    intervener = None
    if collector_type == 'real':
        from mile_franka.collect import Collector, ScriptedIntervener, TeleopIntervener
        from mile_franka.config import StackTaskConfig
        real_collector = Collector(env)
        which = config['experiment'].get('intervener', 'scripted')
        if which == 'scripted':
            from mile_franka.policies.scripted import ScriptedStackPolicy
            intervener = ScriptedIntervener(ScriptedStackPolicy(StackTaskConfig(), mediocre=False))
        elif which == 'spacemouse':
            from mile_franka.teleop.spacemouse import SpaceMouseDevice
            intervener = TeleopIntervener(SpaceMouseDevice())
        else:
            raise ValueError(f'Unknown intervener: {which}')
```

Then replace the collection call inside the loop:

```python
        additional_data, mean_score, mean_success_rate = collect_synthetic_data(env=env,
                                                                                n_episodes=episodes_per_round,
                                                                                cost=trainer.intervention_cost,
                                                                                cdf_scale=trainer.intervention_scale,
                                                                                rollout_policy=trainer.policy,
                                                                                intervention_policy=intervention_policy,
                                                                                mental_model=gt_mental_model)
```

with:

```python
        if collector_type == 'real':
            additional_data, mean_score, mean_success_rate = real_collector.collect_intervention(
                policy=trainer.policy, intervener=intervener, n_episodes=episodes_per_round)
        else:
            from collect_synthetic_interventions import collect_synthetic_data
            additional_data, mean_score, mean_success_rate = collect_synthetic_data(env=env,
                                                                                    n_episodes=episodes_per_round,
                                                                                    cost=trainer.intervention_cost,
                                                                                    cdf_scale=trainer.intervention_scale,
                                                                                    rollout_policy=trainer.policy,
                                                                                    intervention_policy=intervention_policy,
                                                                                    mental_model=gt_mental_model)
```

- [ ] **Step 5: Smoke-check train_mile imports without MetaWorld**

Confirms the lazy import works (the module must import even though it references
`ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE`). Run from `scripts/` so the local imports resolve as
they do at runtime:

```bash
cd /home/ray/mile-code/scripts && /home/ray/miniforge3/envs/mile/bin/python -c "
import train_mile
assert hasattr(train_mile, 'build_franka_or_metaworld_env')
print('train_mile import ok')
"; cd /home/ray/mile-code
```
Expected: prints `train_mile import ok`

- [ ] **Step 6: Commit**

```bash
git add scripts/train_mile.py
git commit -m "Run the Franka path: lazy MetaWorld, env branch, real collector swap"
```

---

## Task 5.4: Franka iterative config + end-to-end sim run (spec §8 gate 6)

**Files:** create `config_franka.json`.

A headless iterative run: Franka fake env, the Phase 3 base policy, the scripted human,
auto-eval **off**, tiny epochs/rounds so the smoke completes fast.

- [ ] **Step 1: Write `config_franka.json`**

```json
{
    "train": {
        "num_epochs": 5,
        "batch_size": 256,
        "lr": 0.0001,
        "lambda1": 1.0,
        "lambda2": 1.0
    },
    "experiment": {
        "name": "mile-franka",
        "mode": "iterative",
        "collector": "real",
        "intervener": "scripted",
        "validate": { "enabled": false, "every_n_epochs": 100 },
        "logging": { "terminal_output_to_txt": false, "log_tb": false, "log_wandb": false },
        "save": {
            "enabled": true,
            "every_n_epochs": 0,
            "on_best_validation": false,
            "on_best_rollout_return": false,
            "on_best_rollout_success_rate": false,
            "outdir": "output_dir/franka"
        },
        "rollout": { "enabled": false, "auto_eval": false, "n_episodes": 5, "every_n_epochs": 100 },
        "env_name": "Franka-Stack-Fake-v0",
        "policy_type": "bc",
        "policy_path": "trained_models/franka/base_policy",
        "mental_model_type": "bc",
        "use_warm_start": false,
        "num_rounds": 2,
        "episodes_per_round": 3,
        "include_offline_dataset": false
    }
}
```

- [ ] **Step 2: Ensure the Phase 3 base policy exists**

The run loads `trained_models/franka/base_policy`. If you have not run Task 3.4 in this
session, run it now:

Run: `/home/ray/miniforge3/envs/mile/bin/python scripts/build_base_policy.py`
Expected: prints the `Saved base policy ...` line.

- [ ] **Step 3: Run the end-to-end iterative training**

Run: `cd /home/ray/mile-code/scripts && /home/ray/miniforge3/envs/mile/bin/python train_mile.py --config ../config_franka.json; cd /home/ray/mile-code`
Expected: prints per-round `Collecting intervention data...` and `Current dataset size: ...`,
completes both rounds with **no autonomous rollout** (no MetaWorld import error, no
`NameError`), and writes `output_dir/franka/policy.zip` + `output_dir/franka/mental_model.zip`.

- [ ] **Step 4: Confirm the trained policy was written and reloads**

Run:
```bash
/home/ray/miniforge3/envs/mile/bin/python -c "
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
p = ActorCriticPolicy.load('output_dir/franka/policy')
a, _ = p.predict(np.zeros(72, dtype=np.float32), deterministic=True)
assert a.shape == (4,)
print('e2e franka run ok')
"
```
Expected: prints `e2e franka run ok`

- [ ] **Step 5: Commit the config**

```bash
git add config_franka.json
git commit -m "Add Franka iterative sim config and verify end-to-end MILE run"
```

---

## Phases 3–5 verification

- [ ] **Step 1: Full headless gate sweep**

Run, in order:
```bash
PY=/home/ray/miniforge3/envs/mile/bin/python
$PY scripts/smoke_franka_env.py                                   # Phase 2 still green
$PY scripts/build_base_policy.py                                  # gate 3
cd /home/ray/mile-code/scripts && $PY train_mile.py --config ../config_franka.json && cd /home/ray/mile-code  # gate 6
```
Expected: each completes without error; the last writes `output_dir/franka/policy.zip`.

- [ ] **Step 2: Confirm git state is clean**

Run: `git status`
Expected: only untracked output artifacts (`output_dir/`, `trained_models/`) remain; all
source committed across Tasks 3.1–5.4.

---

## Self-review (against the spec)

- **Coverage:** scripted policy (mediocre + skilled) → §5.5 ✓ (Task 3.1); BC base policy
  with MILE arch → §5.5/§2 ✓ (Tasks 3.2–3.4); unified Collector with demo-source-agnostic
  interveners + `InterventionDatasetBuilder` + `scripted_intervener` → §5.6 ✓ (Task 4.1);
  `COST_LOOKUP` entry → §5.7 ✓ (Task 5.1); `generate_rollout` guard behind a config flag →
  §5.7/§2 ✓ (Task 5.2); collector swap + `FrameStack+Flatten` on the custom env + Franka path
  not requiring MetaWorld → §5.7 ✓ (Task 5.3); end-to-end iterative N=2, k=3 in sim → §8
  gate 6 ✓ (Task 5.4). Deferred: SpaceMouse live capture (gate 5) is wired (`TeleopIntervener`
  + `intervener: spacemouse`) but exercised in Phase 6/hardware with the device attached;
  MJCF assets + real success-rate-improvement (gates 2 real / 6 real) at sim bring-up
  (Phase 2 checklist); Docker (§6) Phase 6; Vive/AprilTag (§7) Phase 7.
- **Placeholders:** none — every code step is complete and every run step states exact
  expected output. The two tuning notes (Task 3.4 Step 2, success-rate band) give concrete
  knobs, not vague "adjust as needed".
- **Type consistency:** scripted `act(frame(18,)) -> action(4,)`; `Intervener.intervene(obs,
  rollout_action) -> (action(4,), intervene: bool, done: bool)` implemented identically by
  `ScriptedIntervener` and `TeleopIntervener` and consumed once in `Collector`;
  `Collector.collect_intervention -> (dict, float, float)` matches `collect_synthetic_data`;
  `build_actor_critic_policy(obs_space, act_space)` arch matches `train_mile.py`'s bc mental
  model so `ActorCriticPolicy.load` round-trips; `BOX_KEYS` (Phase 1) is the single schema
  source; obs `(72,)` / action `(4,)` consistent with Phase 2.

## Roadmap (remaining)

- **Phase 6 — Docker**: single image `FROM` hucebot's multipanda controller base +
  `pip install -e .` + pinned MILE stack + `pyspacemouse`; compose for SpaceMouse
  (`/dev/hidraw*`), X11, ROS-DDS, optional GPU. The Franka path already imports without
  MetaWorld (Task 5.3), so the image need not install MetaWorld.
- **Phase 7 — France**: `ViveDevice` (hucebot/vive_controller topics) behind the existing
  `TeleopDevice`/`TeleopIntervener` seam (zero collector changes); `AprilTagPoseSource`
  behind `ObjectPoseSource`; camera calibration; `COST_LOOKUP['Franka-Stack-Sim-v0']` re-tune;
  resolve the Phase 2 `CONFIRM@bringup:` markers on the real Panda.
```