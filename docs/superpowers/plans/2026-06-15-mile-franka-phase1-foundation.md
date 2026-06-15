# MILE-on-Franka — Phase 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is **implementation-focused (no TDD)** per project preference — write the code, smoke-check, commit.

**Goal:** Build the hardware-independent foundation for running MILE on a Franka: the teleop-device and object-pose abstractions, and the intervention-dataset builder that produces exactly the dict schema MILE's trainer consumes.

**Architecture:** A new `mile_franka/` package sits alongside the existing `mile/` package. Phase 1 contains only pure-Python units — no ROS2, no MuJoCo, no hardware. The SpaceMouse device takes an injectable raw reader so it works on machines without the device; the dataset builder emits the exact `Box` schema from `mile/scripts/collect_synthetic_interventions.py`.

**Tech Stack:** Python 3.10, numpy. (The repo's pinned `gymnasium`/`stable-baselines3`/`imitation` stack is only needed for the optional dataset round-trip sanity check.)

**Spec:** `docs/superpowers/specs/2026-06-15-mile-franka-stacking-design.md`

---

## Scope of this plan

This is **Phase 1 of 7** (roadmap at the end). It delivers the abstractions and the
dataset contract. Phases 2-7 each get their own plan because they require a sim/hardware
test bed.

## File structure (Phase 1)

```
mile_franka/
  __init__.py
  teleop/
    __init__.py
    base.py                   # TeleopReading dataclass + TeleopDevice ABC
    spacemouse.py             # SpaceMouseDevice (injectable raw reader)
  pose/
    __init__.py
    base.py                   # Pose dataclass + ObjectPoseSource ABC
  data/
    __init__.py
    dataset.py                # InterventionDatasetBuilder (MILE Box schema)
```

Responsibilities: `teleop/` = "human input → 4-DoF action + intervene flag";
`pose/` = "where is the object"; `data/` = "accumulate timesteps → MILE dict".

---

## Task 1: Package skeleton

**Files:** create `mile_franka/__init__.py`, `mile_franka/teleop/__init__.py`,
`mile_franka/pose/__init__.py`, `mile_franka/data/__init__.py`.

- [ ] **Step 1: Create the package directories and empty init files**

```bash
mkdir -p mile_franka/teleop mile_franka/pose mile_franka/data
touch mile_franka/__init__.py mile_franka/teleop/__init__.py \
      mile_franka/pose/__init__.py mile_franka/data/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add mile_franka
git commit -m "Add mile_franka package skeleton"
```

---

## Task 2: TeleopDevice abstraction

**Files:** create `mile_franka/teleop/base.py`.

- [ ] **Step 1: Write `mile_franka/teleop/base.py`**

```python
"""Device-agnostic teleoperation interface.

A TeleopDevice converts raw human input into a 4-DoF action and the two control
signals MILE needs: whether the human is intervening, and whether they ended the
episode. SpaceMouse (home lab) and Vive (France) both implement this interface, so
the data collector is identical across labs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TeleopReading:
    """One poll of a teleop device.

    action: float32 array (4,) = [dx, dy, dz, gripper]; gripper in {-1 open, +1 close}.
    intervene: True when the human is currently taking over control (maps to MILE's nu).
    done: True when the human signals the episode should end.
    """

    action: np.ndarray
    intervene: bool
    done: bool


class TeleopDevice(ABC):
    """Abstract teleop device. Implementations poll hardware and return a reading."""

    @abstractmethod
    def read(self) -> TeleopReading:
        """Return the current reading. Must be non-blocking / fast (called per step)."""

    def close(self) -> None:
        """Release any hardware handles. Default is a no-op."""
```

- [ ] **Step 2: Smoke-check the import**

Run: `python -c "from mile_franka.teleop.base import TeleopDevice, TeleopReading; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/teleop/base.py
git commit -m "Add TeleopDevice abstraction and TeleopReading"
```

---

## Task 3: SpaceMouseDevice

**Files:** create `mile_franka/teleop/spacemouse.py`.

`intervene` is true when translation exceeds the deadband OR the close-gripper button is
held — mirroring the clutch semantics of the Vive teleop bridge used in France. The raw
reader is injectable so the module loads and runs on machines without the device.

- [ ] **Step 1: Write `mile_franka/teleop/spacemouse.py`**

```python
"""SpaceMouse teleop device for the home-lab development setup.

The raw-reader is injectable so the mapping logic runs without hardware. By default it
uses the `pyspacemouse` library; the import is lazy so the module loads on machines
without the device.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mile_franka.teleop.base import TeleopDevice, TeleopReading


class SpaceMouseDevice(TeleopDevice):
    """Maps a 3Dconnexion SpaceMouse to a 4-DoF [dx, dy, dz, gripper] action.

    Args:
        translation_scale: meters of EE delta per unit of puck deflection.
        deadband: puck deflection below this (abs) is treated as zero.
        reader: zero-arg callable returning an object with .x, .y, .z, .buttons.
                Defaults to a pyspacemouse-backed reader.
    """

    def __init__(self, translation_scale: float = 0.02, deadband: float = 0.1,
                 reader: Optional[Callable[[], object]] = None):
        self.translation_scale = translation_scale
        self.deadband = deadband
        self._reader = reader if reader is not None else self._default_reader()

    @staticmethod
    def _default_reader() -> Callable[[], object]:
        import pyspacemouse  # lazy: only needed with real hardware

        pyspacemouse.open()
        return pyspacemouse.read

    def read(self) -> TeleopReading:
        s = self._reader()
        trans = np.array([s.x, s.y, s.z], dtype=np.float32)
        trans[np.abs(trans) < self.deadband] = 0.0
        action_xyz = trans * self.translation_scale

        buttons = tuple(getattr(s, "buttons", (0, 0)))
        gripper_close = bool(buttons[0]) if len(buttons) > 0 else False
        done = bool(buttons[1]) if len(buttons) > 1 else False

        gripper = 1.0 if gripper_close else -1.0
        moved = bool(np.any(action_xyz != 0.0))
        intervene = moved or gripper_close

        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], gripper],
                          dtype=np.float32)
        return TeleopReading(action=action, intervene=intervene, done=done)

    def close(self) -> None:
        try:
            import pyspacemouse

            pyspacemouse.close()
        except Exception:
            pass
```

- [ ] **Step 2: Smoke-check the mapping with a fake reader**

Run:
```bash
python -c "
from dataclasses import dataclass
from mile_franka.teleop.spacemouse import SpaceMouseDevice
@dataclass
class S:
    x: float = 0.0; y: float = 0.0; z: float = 0.0; buttons: tuple = (0, 0)
d = SpaceMouseDevice(translation_scale=0.02, deadband=0.1, reader=lambda: S(x=0.5, buttons=(1,0)))
r = d.read()
assert r.intervene is True and abs(r.action[0]-0.01) < 1e-6 and r.action[3] == 1.0
d2 = SpaceMouseDevice(reader=lambda: S(x=0.05))
assert d2.read().intervene is False
print('spacemouse ok')
"
```
Expected: prints `spacemouse ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/teleop/spacemouse.py
git commit -m "Add SpaceMouseDevice"
```

---

## Task 4: ObjectPoseSource abstraction

**Files:** create `mile_franka/pose/base.py`.

- [ ] **Step 1: Write `mile_franka/pose/base.py`**

```python
"""Object-pose abstraction.

Implementations: MujocoGtPoseSource (sim), AprilTagPoseSource (real, primary),
FoundationPosePoseSource (real, optional markerless). All return a 7-vector
[x, y, z, qx, qy, qz, qw] so the env observation builder is source-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Pose:
    """6-DoF pose as position + quaternion (x, y, z, qx, qy, qz, qw)."""

    position: Sequence[float]
    orientation: Sequence[float]  # quaternion (qx, qy, qz, qw)

    def to_array(self) -> np.ndarray:
        return np.array([*self.position, *self.orientation], dtype=np.float32)


class ObjectPoseSource(ABC):
    """Returns the current pose of a named object in the robot base frame."""

    @abstractmethod
    def get_pose(self, name: str) -> Pose:
        """Return the latest Pose for the object identified by `name`."""
```

- [ ] **Step 2: Smoke-check**

Run:
```bash
python -c "
from mile_franka.pose.base import Pose, ObjectPoseSource
import numpy as np
a = Pose((1,2,3),(0,0,0,1)).to_array()
assert a.shape == (7,) and a.dtype == np.float32
try:
    ObjectPoseSource()  # abstract
    raise SystemExit('should have failed')
except TypeError:
    print('pose ok')
"
```
Expected: prints `pose ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/pose/base.py
git commit -m "Add ObjectPoseSource abstraction and Pose type"
```

---

## Task 5: InterventionDatasetBuilder (locks the MILE schema contract)

**Files:** create `mile_franka/data/dataset.py`.

This guarantees a real human-in-the-loop collector produces exactly the dict that MILE's
continuous (`Box`) trainer path consumes (spec §2). `intervention_prob` is a placeholder
`[1-nu, nu]`: the trainer reads it but overwrites it (`mile/algorithm.py:205`), so only
its presence and shape matter.

- [ ] **Step 1: Write `mile_franka/data/dataset.py`**

```python
"""Builder for MILE intervention datasets (continuous / Box action spaces).

Accumulates per-timestep records and emits the exact dict schema that
mile/scripts/collect_synthetic_interventions.py produces and InterventionTrainer
consumes.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

BOX_KEYS = [
    "state",
    "rollout_action",
    "action",
    "intervention_prob",
    "intervention",
    "reward",
    "next_state",
    "done",
]


class InterventionDatasetBuilder:
    """Accumulate (s, a_r, a_h, nu, r, s', done) records into the MILE Box dict."""

    def __init__(self) -> None:
        self._d: Dict[str, List] = {k: [] for k in BOX_KEYS}

    def add(self, *, state, rollout_action, action, intervention,
            reward, next_state, done) -> None:
        nu = int(bool(intervention))
        self._d["state"].append(np.asarray(state, dtype=np.float32))
        self._d["rollout_action"].append(np.asarray(rollout_action, dtype=np.float32))
        self._d["action"].append(np.asarray(action, dtype=np.float32))
        self._d["intervention_prob"].append(
            np.array([1.0 - nu, float(nu)], dtype=np.float32)
        )
        self._d["intervention"].append(nu)
        self._d["reward"].append(float(reward))
        self._d["next_state"].append(np.asarray(next_state, dtype=np.float32))
        self._d["done"].append(bool(done))

    def to_dict(self) -> Dict[str, List]:
        """Return a shallow copy of the accumulated dict (lists per key)."""
        return {k: list(v) for k, v in self._d.items()}

    def __len__(self) -> int:
        return len(self._d["state"])
```

- [ ] **Step 2: Smoke-check schema + round-trip through the real MILE pipeline**

Requires the `mile` package installed (`pip install -e .`, per CLAUDE.md).

Run:
```bash
python -c "
import numpy as np
from mile_franka.data.dataset import InterventionDatasetBuilder, BOX_KEYS
from mile.utils import prepare_dataset, DictDataset
b = InterventionDatasetBuilder()
for i in range(10):
    b.add(state=np.zeros(8), rollout_action=np.zeros(4), action=np.ones(4),
          intervention=i % 2, reward=0.0, next_state=np.zeros(8), done=False)
assert len(b) == 10
d = b.to_dict()
assert np.allclose(d['intervention_prob'][1], [0.0, 1.0])  # nu=1 row
train, valid = prepare_dataset(d, ratio=0.8)
ds = DictDataset(train)
assert len(ds) == 8 and set(ds[0].keys()) == set(BOX_KEYS)
print('dataset ok')
"
```
Expected: prints `dataset ok`

- [ ] **Step 3: Commit**

```bash
git add mile_franka/data/dataset.py
git commit -m "Add InterventionDatasetBuilder matching MILE Box schema"
```

---

## Task 6: Phase 1 verification

- [ ] **Step 1: Confirm the whole package imports**

Run:
```bash
python -c "import mile_franka.teleop.base, mile_franka.teleop.spacemouse, mile_franka.pose.base, mile_franka.data.dataset; print('phase1 ok')"
```
Expected: prints `phase1 ok`

- [ ] **Step 2: Confirm git state is clean**

Run: `git status`
Expected: nothing to commit (all Phase 1 work committed across Tasks 1-5).

---

## Self-review (against the spec)

- **Coverage:** `TeleopDevice`/`SpaceMouseDevice` → §5.3 ✓; `ObjectPoseSource`/`Pose`
  → §5.4 ✓; dataset schema + `intervention_prob` placeholder → §2 / §5.6 ✓. FrankaEnv,
  scripted/BC, collector, train integration, assets, Docker, France deferred to Phases 2-7.
- **Placeholders:** none — every step has complete code and exact expected output.
- **Type consistency:** `TeleopReading.action` float32 (4,); `Pose.to_array` float32 (7,);
  `BOX_KEYS` defined once in `data/dataset.py` and imported elsewhere (no drift).

## Roadmap (subsequent plans, not tasks)

Each gets its own implementation-focused plan when started (each needs a sim/hardware bed):

- **Phase 2 — `FrankaEnv`** (`mile_franka/envs/franka_env.py`): rclpy node →
  `/cartesian_impedance/equilibrium_pose` + FrankaState + gripper action server;
  `MujocoGtPoseSource`; `FrameStack(4)+Flatten`; `info['success']`; gym registration.
- **Phase 3 — scripted stack policy + BC base policy**: `scripted_stack_policy` (GT poses),
  demo generation, imitation BC → mediocre `ActorCriticPolicy`.
- **Phase 4 — Collector** (`mile_franka/collect.py`): unified `mode={'demo','intervention'}`
  using `InterventionDatasetBuilder`; `scripted_intervener` for headless runs.
- **Phase 5 — train_mile integration**: swap synthetic → real collector; add `COST_LOOKUP`
  entry; guard the auto `generate_rollout` (algorithm.py:186) behind a config flag;
  iterative N=2-3, k=3 in sim.
- **Phase 6 — Docker**: single image `FROM` the multipanda base + pinned MILE stack +
  `pyspacemouse`; compose for SpaceMouse/X11/ROS-DDS/GPU passthrough.
- **Phase 7 — France**: `ViveDevice` (hucebot/vive_controller topics), `AprilTagPoseSource`,
  camera calibration, cost re-tune.
