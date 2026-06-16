# MILE-on-Franka — Phase 2: FrankaEnv Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is **implementation-focused (no TDD)** per project preference — write the code, smoke-check, commit.

**Goal:** Build `FrankaEnv` — a `gymnasium.Env` that drives the block-stacking task over the multipanda_ros2 Cartesian-impedance interface, exposing the 4-DoF action space, the state observation, and `info['success']` that MILE's `generate_rollout` requires — with the ROS2/MuJoCo coupling isolated behind an injectable `RobotBackend` so the env logic is fully testable headless.

**Architecture:** `FrankaEnv` depends on two abstractions: a `RobotBackend` (set equilibrium pose, command gripper, read EE pose/width, reset) and an `ObjectPoseSource` (Phase 1) for cube poses. The home-lab/CI path injects a `FakeRobotBackend` + a shared `FakeWorld` (a minimal pick-and-place kinematic model) and a `WorldPoseSource`, so reset/step/success are verifiable with no ROS or MuJoCo. The France/sim path injects `MultipandaRosBackend` (lazy `rclpy`) + `MujocoGtPoseSource`. Both implement the identical interface, so `FrankaEnv` is backend-agnostic — the same point the spec makes about the controller.

**Tech Stack:** Python 3.10, numpy, `gymnasium==0.29.1` (`FrameStack`, `FlattenObservation`). `rclpy` and `franka_msgs`/`geometry_msgs` are imported lazily only inside `MultipandaRosBackend`/`MujocoGtPoseSource`, so the package imports and the headless smoke checks run without ROS.

**Spec:** `docs/superpowers/specs/2026-06-15-mile-franka-stacking-design.md` (§4 task, §5.2 FrankaEnv, §5.4 ObjectPoseSource, §8 gate 1)

**Builds on:** `docs/superpowers/plans/2026-06-15-mile-franka-phase1-foundation.md` (provides `mile_franka.pose.base.Pose` / `ObjectPoseSource`).

---

## Scope of this plan

This is **Phase 2 of 7**. It delivers a working, headless-testable `FrankaEnv` and its
fake backend, plus the real `MultipandaRosBackend` / `MujocoGtPoseSource` code (which
cannot be exercised without the multipanda sim and so end in a **manual bring-up gate**,
Task 9). It does **not** add the scripted/BC policy (Phase 3), the collector (Phase 4),
or any `train_mile.py` edits (Phase 5).

### What is verifiable now vs. at sim bring-up

- **Now (headless, no ROS):** Tasks 1–6 — config, backend ABC, fake world, `FrankaEnv`
  reset/step/obs/reward/`info['success']`, gym registration + wrappers, and a smoke
  script that hand-drives the fake env through a full stack and asserts success toggles
  (spec §8 gate 1, fake variant).
- **At multipanda sim bring-up (France/lab, has ROS):** Tasks 7–9 — the real backend and
  GT pose source are written now but verified later against the live sim via the manual
  checklist in Task 9. **Open items §10** (down-facing wrist quaternion, exact equilibrium
  topic/gripper-action names, cube-pose mechanism) are confirmed there; the code marks each
  with a single `CONFIRM@bringup:` comment so they are greppable.

> **Controller image — use hucebot's docker (not a hand-rolled stack).** The
> Cartesian-impedance controller, MuJoCo sim, and FrankaState/gripper interfaces this env
> drives are the ones shipped in **hucebot's multipanda_ros2 controller docker** (the
> image the France lab runs). All `CONFIRM@bringup:` topic/action/frame names must be read
> off a container started from **that** image — `ros2 topic list` / `ros2 action list`
> against the running hucebot controller — not assumed. `FrankaEnv` and
> `MultipandaRosBackend` run as a ROS2 client that joins the same DDS graph as that
> container (shared `ROS_DOMAIN_ID` / host networking). This is also the base image Phase 6
> layers the MILE stack onto, so the controller is identical in the lab and in France.

> **Docker-readiness — everything here must ship in the Phase 6 image.** Three invariants
> the tasks below uphold so Phase 6 can `pip install` this package onto hucebot's base
> image with no surprises:
> 1. **`mile_franka` is an installable package.** `setup.py` currently only ships `mile`
>    (and Phase 1 never fixed it), so an image built today would `import mile_franka` fail.
>    **Task 9 makes the whole package + subpackages installable** and the smoke scripts
>    runnable from an installed copy (not just the repo cwd).
> 2. **No import-time ROS/MuJoCo dependency.** `rclpy`/`franka_msgs`/`geometry_msgs` are
>    imported lazily (Tasks 7–8), so the image builds and the headless gate runs even in a
>    build stage before the ROS layer is active.
> 3. **No new heavyweight pip deps.** Tasks 1–6 add only `numpy` + `gymnasium` (already in
>    `requirements.txt`); nothing here pulls MetaWorld/`mujoco<3`, keeping the Franka path
>    off the conflicting sim stack (spec §6, §9). New runtime deps (`pyspacemouse`,
>    AprilTag) are introduced by their own phases, not here.

## File structure (Phase 2)

```
mile_franka/
  config.py                 # StackTaskConfig dataclass + DOWN_QUAT (shared sim/real dims)
  envs/
    __init__.py
    backend.py              # RobotBackend ABC
    fake_backend.py         # FakeWorld + FakeRobotBackend + WorldPoseSource
    franka_env.py           # FrankaEnv(gym.Env): obs/action spaces, step, reset, reward, success
    registration.py         # register_franka_envs() + make_franka_env() (FrameStack(4)+Flatten)
    ros_backend.py          # MultipandaRosBackend (lazy rclpy) — real, bring-up-gated
  pose/
    mujoco_gt.py            # MujocoGtPoseSource (lazy rclpy PoseStamped subscriber) — bring-up-gated
scripts/
  smoke_franka_env.py       # hand-drives the fake env to a successful stack (gate 1)
```

Responsibilities: `config.py` = "one source of cube/workspace/threshold numbers";
`envs/backend.py` = "the control surface FrankaEnv needs"; `envs/fake_backend.py` =
"a no-ROS pick-and-place world for tests"; `envs/franka_env.py` = "task logic: deltas →
target, obs, reward, success"; `envs/registration.py` = "gym id + wrapper application";
`envs/ros_backend.py` / `pose/mujoco_gt.py` = "the real multipanda coupling".

### Observation / action conventions (locked here, consumed by Phases 3–5)

- **Action:** `Box([-1,-1,-1,-1], [1,1,1,1])`, `float32`, `[dx, dy, dz, gripper]`.
  `dx,dy,dz` are unit-scaled deltas multiplied by `config.action_scale` (meters) and
  integrated into the Cartesian target; `gripper > 0` closes, else opens. Matches the
  MetaWorld 4-DoF `[-1,1]` convention so policies transfer.
- **Observation:** `Box(-inf, inf, (18,))`, `float32` =
  `[ee_xyz(3), gripper_width(1), top_cube_pose(7), bottom_cube_pose(7)]`, where each cube
  pose is `Pose.to_array()` = `[x,y,z,qx,qy,qz,qw]`. EE orientation is omitted (wrist is
  fixed down-facing). After `FrameStack(4)+FlattenObservation` the policy sees `(72,)`.
- **`info['success']`** is `1`/`0` every step (`generate_rollout` reads it,
  `mile/algorithm.py:97`). `terminated = bool(success)`; `truncated = step >= max_steps`.

---

## Task 1: Shared task config

**Files:** create `mile_franka/config.py`.

One dataclass holds every number that must agree between sim MJCF and real cubes (spec
§4, §5.4, §5.8). `DOWN_QUAT` is the fixed down-facing wrist orientation in `(qx,qy,qz,qw)`.

- [ ] **Step 1: Write `mile_franka/config.py`**

```python
"""Shared block-stacking task configuration.

One source of truth for object dimensions, workspace bounds, control scaling, and
success thresholds so the MuJoCo scene (sim) and the real cubes/controller stay
consistent (spec sec 4, 5.4, 5.8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fixed down-facing wrist orientation as a quaternion (qx, qy, qz, qw): a 180-deg
# rotation about +X points the gripper at the table. CONFIRM@bringup: the exact quat
# the multipanda Cartesian-impedance controller expects for "pointing down".
DOWN_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class StackTaskConfig:
    """Numbers shared across sim and real for the cube-stacking task.

    Lengths are meters in the robot base frame.
    """

    cube_size: float = 0.04                  # cube edge length (< gripper max open)
    gripper_open_width: float = 0.08         # finger separation when open (m)
    gripper_closed_width: float = 0.0        # finger separation when fully closed (m)

    # Axis-aligned workspace box the EE target is clipped into: [low, high] per axis.
    workspace_low: np.ndarray = field(
        default_factory=lambda: np.array([0.30, -0.30, 0.02], dtype=np.float32))
    workspace_high: np.ndarray = field(
        default_factory=lambda: np.array([0.70, 0.30, 0.40], dtype=np.float32))
    table_z: float = 0.02                    # z of the table surface (cube resting plane)

    action_scale: float = 0.05               # meters of EE delta per unit action component
    max_steps: int = 150                     # truncation horizon

    # Success: top cube centered on bottom cube and released.
    success_xy_tol: float = 0.02             # max horizontal center offset (m)
    success_z_tol: float = 0.01              # max error in stacked height (m)

    # Reset randomization: cube centers drawn from an inner margin of the workspace.
    reset_margin: float = 0.05               # m kept clear of workspace xy edges
    reset_min_separation: float = 0.10       # m minimum xy gap between the two cubes
```

- [ ] **Step 2: Smoke-check the import and defaults**

Run:
```bash
python -c "
import numpy as np
from mile_franka.config import StackTaskConfig, DOWN_QUAT
c = StackTaskConfig()
assert c.workspace_low.shape == (3,) and c.workspace_high.shape == (3,)
assert c.cube_size < c.gripper_open_width
assert DOWN_QUAT.shape == (4,)
print('config ok')
"
```
Expected: prints `config ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/config.py
git commit -m "Add StackTaskConfig and DOWN_QUAT for the stacking task"
```

---

## Task 2: RobotBackend abstraction

**Files:** create `mile_franka/envs/__init__.py`, `mile_franka/envs/backend.py`.

This is the only control surface `FrankaEnv` uses. Sim and real both implement it; the
env never imports `rclpy`.

- [ ] **Step 1: Create the envs package init**

```bash
touch mile_franka/envs/__init__.py
```

- [ ] **Step 2: Write `mile_franka/envs/backend.py`**

```python
"""The robot control surface FrankaEnv depends on.

Implementations: FakeRobotBackend (headless tests) and MultipandaRosBackend (real
multipanda_ros2 sim/hardware). FrankaEnv talks only to this interface, so it never
imports rclpy and is identical across sim and hardware (spec sec 5.2).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RobotBackend(ABC):
    """Cartesian-impedance + gripper control surface, in the robot base frame."""

    @abstractmethod
    def reset(self, np_random: np.random.Generator) -> None:
        """Bring the robot/scene to a start state. Sim randomizes; real is human-gated."""

    @abstractmethod
    def set_equilibrium_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Command the Cartesian equilibrium target: position (3,) + quaternion (4,) xyzw."""

    @abstractmethod
    def set_gripper(self, command: float) -> None:
        """Drive the gripper: command > 0 closes, command <= 0 opens."""

    @abstractmethod
    def get_ee_position(self) -> np.ndarray:
        """Return the current end-effector position (3,) float32."""

    @abstractmethod
    def get_gripper_width(self) -> float:
        """Return the current finger separation in meters."""

    def close(self) -> None:
        """Release any hardware/ROS handles. Default is a no-op."""
```

- [ ] **Step 3: Smoke-check**

Run:
```bash
python -c "
from mile_franka.envs.backend import RobotBackend
try:
    RobotBackend()  # abstract
    raise SystemExit('should have failed')
except TypeError:
    print('backend ok')
"
```
Expected: prints `backend ok`

- [ ] **Step 4: Commit**

```bash
git add mile_franka/envs/__init__.py mile_franka/envs/backend.py
git commit -m "Add RobotBackend abstraction"
```

---

## Task 3: FakeWorld + FakeRobotBackend + WorldPoseSource

**Files:** create `mile_franka/envs/fake_backend.py`.

A minimal kinematic pick-and-place world with no ROS/MuJoCo. EE tracks the commanded
target instantly; closing the gripper near the top cube grasps it; the grasped cube
follows the EE; opening while aligned over the bottom cube snaps the top cube onto it
(otherwise it drops to the table). `FakeRobotBackend` and `WorldPoseSource` share one
`FakeWorld`, so `FrankaEnv` sees a consistent control + perception picture — enough to
make `info['success']` toggle under hand-driven actions (spec §8 gate 1).

- [ ] **Step 1: Write `mile_franka/envs/fake_backend.py`**

```python
"""A no-ROS pick-and-place world for headless FrankaEnv tests.

FakeWorld is a tiny kinematic model: the EE teleports to the commanded target, the
gripper grasps the top cube when closed nearby, a grasped cube follows the EE, and
releasing it aligned over the bottom cube stacks it. FakeRobotBackend (control) and
WorldPoseSource (perception) share one FakeWorld instance.
"""
from __future__ import annotations

import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.backend import RobotBackend
from mile_franka.pose.base import ObjectPoseSource, Pose

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)  # cubes stay axis-aligned in the fake world

TOP_CUBE = "top_cube"
BOTTOM_CUBE = "bottom_cube"


class FakeWorld:
    """Shared mutable state for the fake backend and its pose source."""

    def __init__(self, config: StackTaskConfig):
        self.config = config
        self.ee_pos = np.zeros(3, dtype=np.float32)
        self.gripper_width = float(config.gripper_open_width)
        self.grasping = False
        self.top_cube = np.zeros(3, dtype=np.float32)
        self.bottom_cube = np.zeros(3, dtype=np.float32)

    def reset(self, np_random: np.random.Generator) -> None:
        c = self.config
        lo = c.workspace_low + c.reset_margin
        hi = c.workspace_high - c.reset_margin
        rest_z = c.table_z + c.cube_size / 2.0
        # Draw two xy centers at least reset_min_separation apart.
        while True:
            bottom_xy = np_random.uniform(lo[:2], hi[:2])
            top_xy = np_random.uniform(lo[:2], hi[:2])
            if np.linalg.norm(top_xy - bottom_xy) >= c.reset_min_separation:
                break
        self.bottom_cube = np.array([*bottom_xy, rest_z], dtype=np.float32)
        self.top_cube = np.array([*top_xy, rest_z], dtype=np.float32)
        # Start the EE above the top cube, gripper open, nothing grasped.
        self.ee_pos = np.array([*top_xy, hi[2]], dtype=np.float32)
        self.gripper_width = float(c.gripper_open_width)
        self.grasping = False

    def set_ee(self, position: np.ndarray) -> None:
        self.ee_pos = np.asarray(position, dtype=np.float32).copy()
        if self.grasping:
            self.top_cube = self.ee_pos.copy()  # grasped cube sits at the gripper

    def set_gripper(self, command: float) -> None:
        c = self.config
        if command > 0:  # close
            self.gripper_width = float(c.gripper_closed_width)
            if not self.grasping and np.linalg.norm(self.ee_pos - self.top_cube) < c.cube_size:
                self.grasping = True
                self.top_cube = self.ee_pos.copy()
        else:  # open / release
            self.gripper_width = float(c.gripper_open_width)
            if self.grasping:
                self.grasping = False
                rest_z = c.table_z + c.cube_size / 2.0
                xy_off = np.linalg.norm(self.top_cube[:2] - self.bottom_cube[:2])
                if xy_off < c.success_xy_tol:  # aligned: snap onto the bottom cube
                    self.top_cube = np.array(
                        [*self.bottom_cube[:2], rest_z + c.cube_size], dtype=np.float32)
                else:  # misaligned: drop to the table under the EE
                    self.top_cube = np.array(
                        [self.ee_pos[0], self.ee_pos[1], rest_z], dtype=np.float32)


class FakeRobotBackend(RobotBackend):
    """RobotBackend backed by a FakeWorld (no ROS)."""

    def __init__(self, world: FakeWorld):
        self.world = world

    def reset(self, np_random: np.random.Generator) -> None:
        self.world.reset(np_random)

    def set_equilibrium_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        self.world.set_ee(position)  # orientation ignored: fake wrist is always down

    def set_gripper(self, command: float) -> None:
        self.world.set_gripper(command)

    def get_ee_position(self) -> np.ndarray:
        return self.world.ee_pos.copy()

    def get_gripper_width(self) -> float:
        return float(self.world.gripper_width)


class WorldPoseSource(ObjectPoseSource):
    """ObjectPoseSource reading cube poses from a shared FakeWorld."""

    def __init__(self, world: FakeWorld):
        self.world = world

    def get_pose(self, name: str) -> Pose:
        if name == TOP_CUBE:
            return Pose(tuple(self.world.top_cube), IDENTITY_QUAT)
        if name == BOTTOM_CUBE:
            return Pose(tuple(self.world.bottom_cube), IDENTITY_QUAT)
        raise KeyError(f"unknown object: {name!r}")
```

- [ ] **Step 2: Smoke-check grasp → align → release stacks the cube**

Run:
```bash
python -c "
import numpy as np
from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import FakeWorld, FakeRobotBackend, WorldPoseSource, TOP_CUBE, BOTTOM_CUBE
c = StackTaskConfig()
w = FakeWorld(c); w.reset(np.random.default_rng(0))
be = FakeRobotBackend(w); ps = WorldPoseSource(w)
# Move onto the top cube and close -> grasp.
be.set_equilibrium_pose(w.top_cube.copy(), None); be.set_gripper(1.0)
assert w.grasping is True
# Carry it over the bottom cube xy and release -> stacked one cube_size up.
target = np.array([w.bottom_cube[0], w.bottom_cube[1], 0.2], dtype=np.float32)
be.set_equilibrium_pose(target, None); be.set_gripper(-1.0)
top = ps.get_pose(TOP_CUBE).to_array(); bot = ps.get_pose(BOTTOM_CUBE).to_array()
assert np.linalg.norm(top[:2]-bot[:2]) < c.success_xy_tol
assert abs(top[2]-bot[2]-c.cube_size) < 1e-5
print('fakeworld ok')
"
```
Expected: prints `fakeworld ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/envs/fake_backend.py
git commit -m "Add FakeWorld pick-and-place backend and WorldPoseSource"
```

---

## Task 4: FrankaEnv

**Files:** create `mile_franka/envs/franka_env.py`.

The task logic: integrate the action delta into a clipped Cartesian target, command the
backend, read EE + gripper, query cube poses, build the 18-vector obs, compute reward and
`info['success']`. Backend-agnostic.

- [ ] **Step 1: Write `mile_franka/envs/franka_env.py`**

```python
"""FrankaEnv: gymnasium env for block stacking over a RobotBackend (spec sec 5.2).

Action (4,) [dx, dy, dz, gripper] in [-1, 1]: xyz are action_scale-meter deltas
integrated into the Cartesian target; gripper > 0 closes. Observation (18,) =
[ee_xyz(3), gripper_width(1), top_cube_pose(7), bottom_cube_pose(7)]. info['success']
is 1 when the top cube rests centered on the bottom cube with the gripper released.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from mile_franka.config import DOWN_QUAT, StackTaskConfig
from mile_franka.envs.backend import RobotBackend
from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE
from mile_franka.pose.base import ObjectPoseSource


class FrankaEnv(gym.Env):
    """Block-stacking environment over an injected RobotBackend + ObjectPoseSource."""

    metadata = {"render_modes": []}

    def __init__(self, backend: RobotBackend, pose_source: ObjectPoseSource,
                 config: Optional[StackTaskConfig] = None, mode: str = "sim"):
        super().__init__()
        self.backend = backend
        self.pose_source = pose_source
        self.config = config if config is not None else StackTaskConfig()
        self.mode = mode

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        self._ee_target = np.zeros(3, dtype=np.float32)
        self._step_count = 0

    def _build_obs(self) -> np.ndarray:
        ee = np.asarray(self.backend.get_ee_position(), dtype=np.float32)
        width = np.array([self.backend.get_gripper_width()], dtype=np.float32)
        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        return np.concatenate([ee, width, top, bottom]).astype(np.float32)

    def _is_success(self) -> bool:
        c = self.config
        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        xy_off = float(np.linalg.norm(top[:2] - bottom[:2]))
        target_z = float(bottom[2]) + c.cube_size
        z_err = abs(float(top[2]) - target_z)
        released = self.backend.get_gripper_width() > c.gripper_closed_width + 1e-4
        return xy_off < c.success_xy_tol and z_err < c.success_z_tol and released

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.backend.reset(self.np_random)
        self._ee_target = np.asarray(
            self.backend.get_ee_position(), dtype=np.float32).copy()
        self._step_count = 0
        return self._build_obs(), {"success": 0}

    def step(self, action):
        c = self.config
        action = np.asarray(action, dtype=np.float32).reshape(4)
        self._ee_target = np.clip(
            self._ee_target + action[:3] * c.action_scale,
            c.workspace_low, c.workspace_high).astype(np.float32)
        self.backend.set_equilibrium_pose(self._ee_target, DOWN_QUAT)
        self.backend.set_gripper(float(action[3]))

        obs = self._build_obs()
        success = self._is_success()

        top = self.pose_source.get_pose(TOP_CUBE).to_array()
        bottom = self.pose_source.get_pose(BOTTOM_CUBE).to_array()
        xy_off = float(np.linalg.norm(top[:2] - bottom[:2]))
        z_err = abs(float(top[2]) - (float(bottom[2]) + c.cube_size))
        reward = -xy_off - z_err + (10.0 if success else 0.0)

        self._step_count += 1
        terminated = bool(success)
        truncated = self._step_count >= c.max_steps
        return obs, float(reward), terminated, truncated, {"success": int(success)}

    def close(self):
        self.backend.close()
```

- [ ] **Step 2: Smoke-check reset/step shapes and the success flag**

Run:
```bash
python -c "
import numpy as np
from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import FakeWorld, FakeRobotBackend, WorldPoseSource
from mile_franka.envs.franka_env import FrankaEnv
c = StackTaskConfig()
w = FakeWorld(c)
env = FrankaEnv(FakeRobotBackend(w), WorldPoseSource(w), c)
obs, info = env.reset(seed=0)
assert obs.shape == (18,) and obs.dtype == np.float32 and info['success'] == 0
obs, r, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
assert obs.shape == (18,) and 'success' in info and isinstance(r, float)
assert term is False
print('frankaenv ok')
"
```
Expected: prints `frankaenv ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/envs/franka_env.py
git commit -m "Add FrankaEnv block-stacking gym environment"
```

---

## Task 5: Gym registration + wrapper helper

**Files:** create `mile_franka/envs/registration.py`.

`make_franka_env` builds a `FrankaEnv` and applies `FrameStack(4)+FlattenObservation`
(spec §5.7 — the same wrappers MetaWorld gets, generalized to the custom env). A
registered `Franka-Stack-Fake-v0` id constructs the fake-backed env so `gym.make` works
for headless runs; the real path calls `make_franka_env` with injected backends.

- [ ] **Step 1: Write `mile_franka/envs/registration.py`**

```python
"""Gym registration and the FrameStack(4)+Flatten wrapper helper (spec sec 5.7).

make_franka_env wraps a FrankaEnv exactly like train_mile.py wraps MetaWorld envs, so a
policy sees a (72,) observation (4 stacked 18-vectors). Franka-Stack-Fake-v0 builds the
no-ROS fake-backed env for headless development.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack

from mile_franka.config import StackTaskConfig
from mile_franka.envs.fake_backend import FakeRobotBackend, FakeWorld, WorldPoseSource
from mile_franka.envs.franka_env import FrankaEnv

FAKE_ENV_ID = "Franka-Stack-Fake-v0"


def _build_fake_env(config: Optional[StackTaskConfig] = None) -> FrankaEnv:
    config = config if config is not None else StackTaskConfig()
    world = FakeWorld(config)
    return FrankaEnv(FakeRobotBackend(world), WorldPoseSource(world), config, mode="sim")


def make_franka_env(env: FrankaEnv, frame_stack: int = 4) -> gym.Env:
    """Apply FrameStack + FlattenObservation to a FrankaEnv (matches the MetaWorld path)."""
    return FlattenObservation(FrameStack(env, frame_stack))


def register_franka_envs() -> None:
    """Register Franka gym ids (idempotent)."""
    if FAKE_ENV_ID in gym.registry:
        return
    gym.register(id=FAKE_ENV_ID, entry_point=_build_fake_env)
```

- [ ] **Step 2: Smoke-check registration + wrapped observation shape**

Run:
```bash
python -c "
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack
from mile_franka.envs.registration import register_franka_envs, FAKE_ENV_ID
register_franka_envs()
env = FlattenObservation(FrameStack(gym.make(FAKE_ENV_ID), 4))
obs, info = env.reset(seed=0)
assert obs.shape == (72,), obs.shape
obs, r, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
assert obs.shape == (72,) and info['success'] in (0, 1)
print('registration ok')
"
```
Expected: prints `registration ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/envs/registration.py
git commit -m "Register Franka-Stack-Fake-v0 and add FrameStack+Flatten helper"
```

---

## Task 6: Headless success smoke script (spec §8 gate 1, fake variant)

**Files:** create `scripts/smoke_franka_env.py`.

Hand-drives the fake env through a full stack (approach → grasp → lift → carry → release)
using only the public `step` API and asserts `info['success']` flips to 1 — the spec's
acceptance gate 1, runnable with no ROS.

- [ ] **Step 1: Write `scripts/smoke_franka_env.py`**

```python
"""Headless acceptance gate 1 (spec sec 8): drive FrankaEnv to a successful stack.

Uses the fake backend so it runs with no ROS/MuJoCo. Moves the EE with clipped deltas
(the only control FrankaEnv exposes), grasps the top cube, carries it over the bottom
cube, releases, and asserts info['success'] == 1.
"""
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
```

- [ ] **Step 2: Run the smoke script**

Run: `python scripts/smoke_franka_env.py`
Expected: prints `smoke_franka_env ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_franka_env.py
git commit -m "Add headless FrankaEnv stacking smoke script (acceptance gate 1)"
```

---

## Task 7: MultipandaRosBackend (real, bring-up-gated)

**Files:** create `mile_franka/envs/ros_backend.py`.

The real backend over multipanda_ros2: publish `geometry_msgs/PoseStamped` to the
Cartesian-impedance equilibrium topic, read EE pose from the FrankaState broadcaster,
drive the gripper via its action server. `rclpy` and message types are imported lazily so
the package still imports without ROS. **This is verified later in Task 9**, not here —
the three unconfirmed ROS specifics are marked `CONFIRM@bringup:`.

- [ ] **Step 1: Write `mile_franka/envs/ros_backend.py`**

```python
"""Real multipanda_ros2 backend (spec sec 5.2). Verified at sim bring-up (Task 9).

This is a ROS2 client of hucebot's multipanda_ros2 controller docker (the France lab's
image): it joins that container's DDS graph and drives the controller running there. Do
not assume a hand-rolled controller -- read the real topic/action/frame names off the
running hucebot container. rclpy / ROS message imports are lazy so importing this module
never requires ROS. CONFIRM@bringup markers flag the specifics to confirm against that
live container (spec sec 10).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from mile_franka.config import StackTaskConfig
from mile_franka.envs.backend import RobotBackend

# CONFIRM@bringup: exact topic/action names exposed by the multipanda controller.
EQUILIBRIUM_TOPIC = "/cartesian_impedance/equilibrium_pose"
FRANKA_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
GRIPPER_ACTION = "/fr3_gripper/grasp"  # franka_msgs/action/Grasp
CONTROL_PERIOD_S = 0.1  # ~10 Hz control step (spec sec 5.2)


class MultipandaRosBackend(RobotBackend):
    """RobotBackend backed by a live multipanda_ros2 node (sim or hardware)."""

    def __init__(self, config: Optional[StackTaskConfig] = None,
                 base_frame: str = "fr3_link0"):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped

        self.config = config if config is not None else StackTaskConfig()
        self.base_frame = base_frame  # CONFIRM@bringup: robot base frame id

        if not rclpy.ok():
            rclpy.init()
        self._node: Node = rclpy.create_node("mile_franka_backend")
        self._pub = self._node.create_publisher(PoseStamped, EQUILIBRIUM_TOPIC, 10)

        self._ee_position = np.zeros(3, dtype=np.float32)
        self._gripper_width = float(self.config.gripper_open_width)
        self._lock = threading.Lock()
        self._subscribe_state()

        # Spin in a background thread so callbacks fire while the env runs.
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _subscribe_state(self) -> None:
        # CONFIRM@bringup: FrankaState message type + the field holding the EE pose.
        from franka_msgs.msg import FrankaRobotState

        def _on_state(msg) -> None:
            pose = msg.o_t_ee.pose  # CONFIRM@bringup: EE pose accessor
            with self._lock:
                self._ee_position = np.array(
                    [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)

        self._node.create_subscription(
            FrankaRobotState, FRANKA_STATE_TOPIC, _on_state, 10)

    def reset(self, np_random: np.random.Generator) -> None:
        # Real reset is human-gated (spec sec 5.2): place the cubes, then continue.
        input("Place the two cubes in the workspace and press Enter to start the episode...")

    def set_equilibrium_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        self._pub.publish(msg)
        time.sleep(CONTROL_PERIOD_S)  # wait one control period for the controller to track

    def set_gripper(self, command: float) -> None:
        # CONFIRM@bringup: gripper action interface (Grasp vs Move) + width/speed/force.
        from franka_msgs.action import Grasp
        from rclpy.action import ActionClient

        client = ActionClient(self._node, Grasp, GRIPPER_ACTION)
        if not client.wait_for_server(timeout_sec=2.0):
            return
        goal = Grasp.Goal()
        c = self.config
        goal.width = c.gripper_closed_width if command > 0 else c.gripper_open_width
        goal.speed = 0.1
        goal.force = 20.0
        client.send_goal_async(goal)  # fire-and-forget; controller closes/opens
        with self._lock:
            self._gripper_width = float(goal.width)

    def get_ee_position(self) -> np.ndarray:
        with self._lock:
            return self._ee_position.copy()

    def get_gripper_width(self) -> float:
        with self._lock:
            return float(self._gripper_width)

    def close(self) -> None:
        try:
            self._executor.shutdown()
            self._node.destroy_node()
        except Exception:
            pass
```

- [ ] **Step 2: Smoke-check the module imports without ROS installed**

The `rclpy` imports are inside `__init__`/methods, so importing the module must succeed
even where ROS is absent (constructing the backend is deferred to Task 9).

Run:
```bash
python -c "
import mile_franka.envs.ros_backend as rb
assert hasattr(rb, 'MultipandaRosBackend')
print('ros_backend import ok')
"
```
Expected: prints `ros_backend import ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/envs/ros_backend.py
git commit -m "Add MultipandaRosBackend (lazy rclpy, bring-up-gated)"
```

---

## Task 8: MujocoGtPoseSource (real, bring-up-gated)

**Files:** create `mile_franka/pose/mujoco_gt.py`.

Ground-truth cube poses from the multipanda MuJoCo sim, delivered as
`geometry_msgs/PoseStamped` on a per-object topic (same message type the AprilTag and
FoundationPose sources use, spec §5.4 — so the abstraction is uniform). Lazy `rclpy`.
Verified in Task 9.

- [ ] **Step 1: Write `mile_franka/pose/mujoco_gt.py`**

```python
"""Ground-truth cube poses from the multipanda MuJoCo sim (spec sec 5.4).

Subscribes to one geometry_msgs/PoseStamped topic per object (the same message type the
real AprilTag/FoundationPose sources publish), so FrankaEnv's observation builder is
source-agnostic. rclpy is imported lazily; verified at sim bring-up (Task 9).

CONFIRM@bringup: how the multipanda sim exposes GT cube poses (per-object PoseStamped
topic as assumed here, vs direct mjData access) -- spec sec 10.
"""
from __future__ import annotations

import threading
from typing import Dict

import numpy as np

from mile_franka.pose.base import ObjectPoseSource, Pose


class MujocoGtPoseSource(ObjectPoseSource):
    """ObjectPoseSource subscribing to GT PoseStamped topics from the sim."""

    def __init__(self, node, topics: Dict[str, str]):
        """node: an rclpy Node (reuse the backend's). topics: object name -> topic."""
        from geometry_msgs.msg import PoseStamped

        self._lock = threading.Lock()
        self._latest: Dict[str, Pose] = {}
        for name, topic in topics.items():
            node.create_subscription(
                PoseStamped, topic, self._make_cb(name), 10)

    def _make_cb(self, name: str):
        def _cb(msg) -> None:
            p, o = msg.pose.position, msg.pose.orientation
            with self._lock:
                self._latest[name] = Pose((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        return _cb

    def get_pose(self, name: str) -> Pose:
        with self._lock:
            if name not in self._latest:
                raise KeyError(f"no pose received yet for {name!r}")
            return self._latest[name]
```

- [ ] **Step 2: Smoke-check the module imports without ROS installed**

Run:
```bash
python -c "
import mile_franka.pose.mujoco_gt as m
assert hasattr(m, 'MujocoGtPoseSource')
print('mujoco_gt import ok')
"
```
Expected: prints `mujoco_gt import ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/pose/mujoco_gt.py
git commit -m "Add MujocoGtPoseSource (lazy rclpy PoseStamped subscriber)"
```

---

## Task 9: Make `mile_franka` installable (Docker-readiness)

**Files:** modify `setup.py`.

`setup.py` ships only `mile`, so the Phase 6 image would `import mile_franka` fail. Switch
to `find_packages` so `mile`, `mile_franka`, and every subpackage install. This is the one
change that lets the whole stack be `pip install`-ed into hucebot's image at the end.

- [ ] **Step 1: Rewrite `setup.py` to discover all packages**

```python
from setuptools import find_packages, setup

setup(
    name="mile",
    version="0.0.1",
    packages=find_packages(include=["mile", "mile.*", "mile_franka", "mile_franka.*"]),
)
```

- [ ] **Step 2: Reinstall editable and verify `mile_franka` is importable as installed**

Run:
```bash
pip install -e . >/dev/null && python -c "
import mile_franka, mile_franka.envs.franka_env, mile_franka.data.dataset
print('install ok')
"
```
Expected: prints `install ok`

- [ ] **Step 3: Verify the package imports from a non-repo working directory**

Confirms the install does not rely on the repo being the current directory (as it will not
be inside the image).

Run: `cd /tmp && python -c "import mile_franka.envs.registration; print('cwd-independent ok')"`
Expected: prints `cwd-independent ok`

- [ ] **Step 4: Commit**

```bash
git add setup.py
git commit -m "Install mile_franka via find_packages for Docker packaging"
```

---

## Task 10: Phase 2 verification + multipanda bring-up gate

This task has two parts: a headless verification block runnable now, and a manual
checklist to run **once the multipanda MuJoCo sim is available** (the spec §8 gate 1 on
the real backend). Record the checklist outcomes inline when you reach the sim.

- [ ] **Step 1: Confirm the whole package imports (no ROS required)**

Run:
```bash
python -c "
import mile_franka.config
import mile_franka.envs.backend, mile_franka.envs.fake_backend
import mile_franka.envs.franka_env, mile_franka.envs.registration
import mile_franka.envs.ros_backend
import mile_franka.pose.mujoco_gt
print('phase2 imports ok')
"
```
Expected: prints `phase2 imports ok`

- [ ] **Step 2: Re-run the headless acceptance gate**

Run: `python scripts/smoke_franka_env.py`
Expected: prints `smoke_franka_env ok`

- [ ] **Step 3: Confirm git state is clean**

Run: `git status`
Expected: nothing to commit (all Phase 2 work committed across Tasks 1–9).

- [ ] **Step 4: Add the bring-up checklist file**

Create `docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md`:

```markdown
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
```

Then:
```bash
git add docs/superpowers/notes/2026-06-15-phase2-bringup-checklist.md
git commit -m "Add Phase 2 multipanda bring-up checklist"
```

---

## Self-review (against the spec)

- **Coverage:** FrankaEnv `step`/`reset`/delta-integration/`info['success']` → §5.2 ✓
  (Task 4); `FrameStack(4)+Flatten` + gym registration → §5.7/§4 ✓ (Task 5);
  `MujocoGtPoseSource` (sim) + real-source PoseStamped uniformity → §5.4 ✓ (Task 8);
  real Cartesian-impedance/FrankaState/gripper coupling → §5.2 ✓ (Task 7); shared object
  dimensions in one config → §5.4/§5.8 ✓ (Task 1); acceptance gate 1 (success toggles
  under hand-driven actions) → §8 ✓ (Task 6 fake / Task 9 real). Deferred per scope:
  scripted/BC policy (§5.5, Phase 3), collector (§5.6, Phase 4), train edits + `COST_LOOKUP`
  + rollout guard (§5.7, Phase 5), assets MJCF (§5.8, folded into Phase 3/bring-up), Docker
  (§6, Phase 6), Vive/AprilTag (§7, Phase 7). Open items §10 are tracked as the three
  `CONFIRM@bringup:` markers resolved in Task 10. **Docker-readiness** (§6/§9): `mile_franka`
  made installable via `find_packages` (Task 9), lazy ROS imports (Tasks 7–8), and no new
  heavyweight deps — so the whole package ships in the Phase 6 image.
- **Placeholders:** none — every code step is complete; the ROS specifics that genuinely
  cannot be known until sim bring-up are real, runnable defaults tagged `CONFIRM@bringup:`,
  not stubs.
- **Type consistency:** action `Box (4,)` float32 and obs `Box (18,)` float32 → `(72,)`
  after wrappers, used identically in Tasks 4/5/6; `RobotBackend` method names
  (`set_equilibrium_pose`, `set_gripper`, `get_ee_position`, `get_gripper_width`, `reset`,
  `close`) match across `backend.py`, `fake_backend.py`, `ros_backend.py`, and every
  `FrankaEnv` call site; `TOP_CUBE`/`BOTTOM_CUBE` object names defined once in
  `fake_backend.py` and imported by `franka_env.py`/`smoke_franka_env.py`; `Pose.to_array()`
  `(7,)` from Phase 1 used unchanged.

## Roadmap (unchanged from Phase 1)

- **Phase 3 — scripted stack policy + BC base policy** (next): `scripted_stack_policy`
  using GT poses on `FrankaEnv`, demo generation, imitation BC → mediocre `ActorCriticPolicy`;
  includes the `assets/stacking.xml` MJCF the bring-up checklist depends on.
- **Phase 4 — Collector**: unified `mode={'demo','intervention'}` over `FrankaEnv` +
  `InterventionDatasetBuilder` (Phase 1); `scripted_intervener` for headless runs.
- **Phase 5 — train_mile integration**: swap synthetic → real collector; add `COST_LOOKUP`
  entry for the Franka env; guard the auto `generate_rollout` (`mile/algorithm.py:186`)
  behind a config flag; iterative N=2–3, k=3 in sim.
- **Phase 6 — Docker**: single image `FROM` hucebot's multipanda controller base +
  `pip install -e .` (now installs `mile_franka` too, Task 9) + pinned MILE stack; compose
  for SpaceMouse/X11/ROS-DDS/GPU passthrough.
- **Phase 7 — France**: `ViveDevice`, `AprilTagPoseSource`, camera calibration, cost re-tune.
```