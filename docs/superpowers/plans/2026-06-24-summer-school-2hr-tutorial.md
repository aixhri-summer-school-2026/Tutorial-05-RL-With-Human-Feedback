# Summer-School 2-Hour Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce everything needed to run the MILE summer-school tutorial — one docker image that runs MetaWorld + all Franka tiers, a keyboard teleop device, a guarded MILE-loss exercise, reduced tutorial configs, a Drive-free artifact bundle, `tutorial-*` make verbs, and the participant/instructor docs.

**Architecture:** Layer tutorial-only code beside the production stack without modifying the MILE method. New code lives in `mile_franka/teleop/keyboard.py`, `mile_franka/tutorial/`, `config_tutorial_*.json`, `scripts/tutorial_*.py`, `docs/tutorial/`, plus Makefile verbs and docker requirement edits. Production `mile/algorithm.py` is never edited; the loss exercise is injected by monkeypatching the module attribute from a tutorial runner.

**Tech Stack:** Python 3.10, gymnasium 0.29.1, stable-baselines3, imitation, torch, pytest, Docker (FROM hucebot:franka-humble), MuJoCo 2.3.7, MetaWorld v2 commit.

**Reference spec:** `docs/superpowers/specs/2026-06-24-summer-school-2hr-tutorial-design.md`

---

## File structure

| File | Responsibility | Phase |
|---|---|---|
| `docker/requirements-mile.txt` (modify) | pin `mujoco==2.3.7`, add MetaWorld commit | A |
| `docker/Dockerfile` (modify) | import-assert `metaworld`; drop "MetaWorld-free" note | A |
| `mile_franka/viz/mujoco_twin.py` (modify) | mujoco 2.3.7 API compat | A |
| `scripts/view_cubes_mujoco.py` (modify) | mujoco 2.3.7 API compat | A |
| `mile_franka/teleop/keyboard.py` (create) | `KeyboardDevice` teleop | B |
| `tests/test_keyboard_device.py` (create) | unit tests for `KeyboardDevice` | B |
| `scripts/train_mile.py` (modify) | wire `intervener: keyboard` | B |
| `mile_franka/tutorial/__init__.py` (create) | package marker | C |
| `mile_franka/tutorial/loss_exercise.py` (create) | the blanked TODO loss | C |
| `mile_franka/tutorial/_loss_solution.py` (create) | reference loss | C |
| `mile_franka/tutorial/loss_loader.py` (create) | validate + auto-apply | C |
| `tests/test_loss_exercise.py` (create) | loss test (the green-light) | C |
| `scripts/tutorial_train.py` (create) | inject loss, then run training | C |
| `config_tutorial_metaworld.json` (create) | reduced Tier-1 config | D |
| `config_tutorial_franka.json` (create) | reduced Tier-3 config | D |
| `scripts/make_tutorial_artifacts.sh` (create) | one-time bundle assembly | E |
| `scripts/fetch_artifacts.py` (create) | Drive-free fetch + checksum | E |
| `tutorial-artifacts.sha256` (create) | committed checksum | E |
| `scripts/tutorial_check.py` (create) | environment readiness check | F |
| `Makefile` (modify) | `tutorial-*` verbs | F |
| `docs/tutorial/0[0-6]-*.md` (create) | participant + instructor docs | G |

Phases are ordered by dependency: B and C are independent of each other; D depends on B+C; E independent; F depends on B–E; G depends on all.

---

## Phase A — Unified docker image

### Task A1: Pin mujoco 2.3.7 and add MetaWorld to the image requirements

**Files:**
- Modify: `docker/requirements-mile.txt`

- [ ] **Step 1: Edit the requirements**

Change the mujoco line and add MetaWorld. The file currently has `mujoco>=3.1.0,<3.2.0`. Replace it and append the MetaWorld pin (same commit as `requirements.txt`):

```
# MILE training layer installed on top of hucebot:franka-humble.
# Keep the base image's gymnasium==0.29.1 and rclpy intact.
# mujoco 2.3.7 is MetaWorld-v2's native version; the Franka sim loop uses mujoco_ros
# (C++), not this Python package, so the downgrade is safe. The twin viewer is adapted
# to 2.3.7 in Task A3.
mujoco==2.3.7
metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@c822f28f582ba1ad49eb5dcf61016566f28003ba
gymnasium==0.29.1
stable-baselines3<=2.3.2
imitation
wandb
tensorboard
pyspacemouse
hidapi
pupil-apriltags
opencv-python-headless
```

- [ ] **Step 2: Commit**

```bash
git add docker/requirements-mile.txt
git commit -m "build(docker): pin mujoco 2.3.7 and add MetaWorld to the image"
```

### Task A2: Extend the Dockerfile build-time import assert

**Files:**
- Modify: `docker/Dockerfile`

- [ ] **Step 1: Update the header comment and the assert**

Replace the first comment line `# ... MetaWorld-free ...` with a note that MetaWorld is now included, and add `metaworld` to the `python3 -c` import assert. The assert line currently imports `rclpy, torch, stable_baselines3, imitation, gymnasium, mile, mile_franka`. Add `metaworld`:

```dockerfile
 && python3 -c \"import rclpy, torch, stable_baselines3, imitation, gymnasium, metaworld, mile, mile_franka; \
from mujoco_ros_msgs.srv import GetBodyState; \
assert gymnasium.__version__=='0.29.1', gymnasium.__version__; \
import mujoco; assert mujoco.__version__.startswith('2.3'), mujoco.__version__; print('deps OK')\""
```

Change the top comment from `# ... MetaWorld-free ...` to:
```dockerfile
# Single MILE-on-multipanda image: hucebot base + the pinned MILE training stack + MetaWorld.
# Python mujoco is 2.3.7 (MetaWorld-v2 native); the Franka sim uses mujoco_ros (C++).
```

- [ ] **Step 2: Commit**

```bash
git add docker/Dockerfile
git commit -m "build(docker): assert metaworld + mujoco 2.3 import at build time"
```

### Task A3: Adapt the twin viewer to the mujoco 2.3.7 API

**Files:**
- Modify: `mile_franka/viz/mujoco_twin.py:64`
- Modify: `scripts/view_cubes_mujoco.py:89,337`

**Context:** Both files do `import mujoco` / `import mujoco.viewer`. Between mujoco 3.1 and 2.3.7 the load/step/viewer APIs are largely identical (`mujoco.MjModel.from_xml_path`, `mujoco.MjData`, `mujoco.mj_step`, `mujoco.viewer.launch_passive` all exist in 2.3.7). The known break is `mujoco.mj_name2id` signature and any use of `mjtObj` enums introduced after 2.3; verify by grepping.

- [ ] **Step 1: Find mujoco API calls used**

Run: `grep -nE "mujoco\.(mj_|Mj|viewer|mjt)" mile_franka/viz/mujoco_twin.py scripts/view_cubes_mujoco.py`
Expected: a list of every mujoco call. For each, confirm it exists in 2.3.7 (the common ones do). The likely-affected call is `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)` — identical in 2.3.7, so usually no change is needed.

- [ ] **Step 2: Apply any required signature fixes**

If grep shows a call removed in 2.3.7, replace it with the 2.3.7 equivalent. If all calls exist in 2.3.7 (the expected case), make a no-op clarifying comment at each `import mujoco` site:

```python
import mujoco  # twin viewer runs on mujoco 2.3.7 in the tutorial image (see Dockerfile)
```

- [ ] **Step 3: Verify the modules import under 2.3.7 (in-container)**

Run (inside the built image): `python3 -c "import mile_franka.viz.mujoco_twin; print('twin import ok')"`
Expected: `twin import ok` with no ImportError. (If mujoco is not installed on the host, defer this check to Task A4.)

- [ ] **Step 4: Commit**

```bash
git add mile_franka/viz/mujoco_twin.py scripts/view_cubes_mujoco.py
git commit -m "fix(twin): mujoco 2.3.7 API compatibility for the tutorial image"
```

### Task A4: Rebuild and smoke-validate the unified image

**Files:** none (validation only)

- [ ] **Step 1: Build the image**

Run: `make build`
Expected: build completes; the build-time assert prints `deps OK`. If `pull access denied … hucebot:franka-humble`, the local base is missing — build it per README before retrying.

- [ ] **Step 2: Verify both stacks import in the container**

Run:
```bash
make up && docker compose -f docker/docker-compose.yml exec sim \
  python3 -c "import metaworld, mile, mile_franka, mujoco; \
print('metaworld', metaworld.__file__ is not None); print('mujoco', mujoco.__version__)"
```
Expected: prints `metaworld True` and `mujoco 2.3.7`.

- [ ] **Step 3: Smoke the MetaWorld env builds**

Run:
```bash
docker compose -f docker/docker-compose.yml exec sim python3 -c \
"from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as E; \
e=E['peg-insert-side-v2-goal-observable'](); e.reset(); print('obs', e.observation_space.shape)"
```
Expected: prints `obs (39,)` (before frame-stack). No mujoco errors.

- [ ] **Step 4: Commit** (nothing to commit; record result in the PR description / notes)

---

## Phase B — KeyboardDevice teleop

### Task B1: Write the KeyboardDevice unit tests (failing)

**Files:**
- Create: `tests/test_keyboard_device.py`

**Context:** `KeyboardDevice` mirrors `JoystickDevice`: injectable `reader` returning a snapshot with a `.keys` set of currently-pressed key names; segment-toggle `intervene`; gripper toggle; `reset()` and `sync_gripper_state()` so `TeleopIntervener` works unchanged. Default key map: movement `w/s` = +x/−x, `a/d` = +y/−y, `q/e` = +z/−z; clutch = `space`; gripper toggle = `g`; done = `enter`; discard = `backspace`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from mile_franka.teleop.keyboard import KeyboardDevice


def _dev(keys_seq):
    """KeyboardDevice whose reader yields successive frozensets of pressed keys."""
    frames = list(keys_seq)
    state = {"i": 0}

    def reader():
        i = min(state["i"], len(frames) - 1)
        state["i"] += 1

        class S:
            keys = frames[i]
        return S()

    return KeyboardDevice(reader=reader, translation_scale=0.02,
                          hold_confirm_s=0.0, debounce_s=0.0)


def test_movement_keys_map_to_xyz_delta():
    dev = _dev([{"w"}])
    r = dev.read()
    assert r.action[0] > 0 and r.action[1] == 0 and r.action[2] == 0
    assert abs(r.action[0] - 0.02) < 1e-6


def test_opposite_keys_have_opposite_sign():
    assert _dev([{"w"}]).read().action[0] > 0
    assert _dev([{"s"}]).read().action[0] < 0
    assert _dev([{"a"}]).read().action[1] > 0
    assert _dev([{"d"}]).read().action[1] < 0


def test_clutch_space_toggles_intervention_segment():
    dev = _dev([set(), {"space"}, set(), {"space"}, set()])
    assert dev.read().intervene is False      # idle
    assert dev.read().intervene is True       # space pressed -> enter segment
    assert dev.read().intervene is True       # still in segment
    assert dev.read().intervene is False      # space again -> exit segment
    assert dev.read().intervene is False


def test_gripper_toggle_alternates_and_starts_nan():
    dev = _dev([set(), {"g"}, set(), {"g"}, set()])
    assert np.isnan(dev.read().action[3])     # before first toggle: nan passthrough
    assert dev.read().action[3] == 1.0        # g -> close
    assert dev.read().action[3] == 1.0        # held value persists
    assert dev.read().action[3] == -1.0       # g -> open


def test_done_and_discard_keys():
    assert _dev([{"enter"}]).read().done is True
    assert _dev([{"backspace"}]).read().discard is True


def test_reset_clears_segment():
    dev = _dev([{"space"}, set()])
    assert dev.read().intervene is True
    dev.reset()
    assert dev._in_segment is False


def test_sync_gripper_state():
    dev = _dev([set()])
    dev.sync_gripper_state(closed=True)
    assert dev._gripper_state == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_keyboard_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mile_franka.teleop.keyboard'`.

### Task B2: Implement KeyboardDevice

**Files:**
- Create: `mile_franka/teleop/keyboard.py`

- [ ] **Step 1: Write the implementation**

```python
"""Keyboard teleop device — the default device for the summer-school tutorial.

Mirrors JoystickDevice (segment-toggle intervention, gripper toggle, injectable
reader) so TeleopIntervener and the collector work unchanged. The reader returns a
snapshot whose `.keys` is the set of currently-pressed key names. The default reader
lazily imports pygame (already in the image) and opens a small focus window.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

import numpy as np

from mile_franka.teleop.base import TeleopDevice, TeleopReading


class KeyboardDevice(TeleopDevice):
    """Maps keyboard input to a 4-DoF [dx, dy, dz, gripper] action.

    Default map: w/s=+x/-x, a/d=+y/-y, q/e=+z/-z, space=clutch (segment toggle),
    g=gripper toggle, enter=done, backspace=discard. Movement keys produce a fixed
    delta of translation_scale metres per step. Gripper starts as nan (passthrough)
    until the first toggle. hold_confirm_s/debounce_s filter key repeat/bounce on the
    edge-triggered keys (clutch/gripper/done/discard).
    """

    def __init__(self, translation_scale: float = 0.02,
                 key_xplus: str = "w", key_xminus: str = "s",
                 key_yplus: str = "a", key_yminus: str = "d",
                 key_zplus: str = "q", key_zminus: str = "e",
                 clutch_key: str = "space", gripper_key: str = "g",
                 done_key: str = "enter", discard_key: str = "backspace",
                 segment_mode: bool = True, gripper_toggle: bool = True,
                 hold_confirm_s: float = 0.0, debounce_s: float = 0.2,
                 reader: Optional[Callable[[], object]] = None):
        self.translation_scale = translation_scale
        self.key_xplus, self.key_xminus = key_xplus, key_xminus
        self.key_yplus, self.key_yminus = key_yplus, key_yminus
        self.key_zplus, self.key_zminus = key_zplus, key_zminus
        self.clutch_key, self.gripper_key = clutch_key, gripper_key
        self.done_key, self.discard_key = done_key, discard_key
        self.segment_mode = segment_mode
        self.gripper_toggle = gripper_toggle
        self.hold_confirm_s = hold_confirm_s
        self.debounce_s = debounce_s
        self._in_segment = False
        self._gripper_state = -1.0  # open
        self._prev_down: dict = {}
        self._press_start: dict = {}
        self._triggered_this_press: dict = {}
        self._last_trigger: dict = {}
        self._reader = reader if reader is not None else self._default_reader()

    def reset(self) -> None:
        self._in_segment = False

    def sync_gripper_state(self, closed: bool) -> None:
        self._gripper_state = 1.0 if closed else -1.0

    @staticmethod
    def _default_reader() -> Callable[[], object]:
        import pygame  # lazy: only needed for live use

        pygame.init()
        pygame.display.set_mode((240, 80))
        pygame.display.set_caption("MILE keyboard teleop — keep focused")
        names = {
            "w": pygame.K_w, "s": pygame.K_s, "a": pygame.K_a, "d": pygame.K_d,
            "q": pygame.K_q, "e": pygame.K_e, "g": pygame.K_g,
            "space": pygame.K_SPACE, "enter": pygame.K_RETURN,
            "backspace": pygame.K_BACKSPACE,
        }

        class _Snapshot:
            __slots__ = ("keys",)

        def _read() -> object:
            pygame.event.pump()
            pressed = pygame.key.get_pressed()
            snap = _Snapshot()
            snap.keys = {n for n, code in names.items() if pressed[code]}
            return snap

        return _read

    def _confirmed_edge(self, keys: Iterable[str], name: str, now: float) -> bool:
        cur = name in keys
        prev = self._prev_down.get(name, False)
        self._prev_down[name] = cur
        if not prev and cur:
            self._press_start[name] = now
            self._triggered_this_press[name] = False
        if prev and not cur:
            self._press_start.pop(name, None)
        if not cur or self._triggered_this_press.get(name, True):
            return False
        held = now - self._press_start.get(name, now)
        since = now - self._last_trigger.get(name, -999.0)
        if held >= self.hold_confirm_s and since >= self.debounce_s:
            self._last_trigger[name] = now
            self._triggered_this_press[name] = True
            return True
        return False

    def read(self) -> TeleopReading:
        now = time.monotonic()
        keys = set(getattr(self._reader(), "keys", ()))

        dx = (self.key_xplus in keys) - (self.key_xminus in keys)
        dy = (self.key_yplus in keys) - (self.key_yminus in keys)
        dz = (self.key_zplus in keys) - (self.key_zminus in keys)
        action_xyz = np.array([dx, dy, dz], dtype=np.float32) * self.translation_scale

        clutch_edge = self._confirmed_edge(keys, self.clutch_key, now)
        gripper_edge = self._confirmed_edge(keys, self.gripper_key, now)
        done = self._confirmed_edge(keys, self.done_key, now)
        discard = self._confirmed_edge(keys, self.discard_key, now)

        if discard:
            self._in_segment = False

        if self.segment_mode:
            if clutch_edge:
                self._in_segment = not self._in_segment
            intervene = self._in_segment
        else:
            intervene = (self.clutch_key in keys) or bool(np.any(action_xyz != 0.0))

        if self.gripper_toggle:
            if gripper_edge:
                self._gripper_state = 1.0 if self._gripper_state < 0 else -1.0
            gripper = self._gripper_state if self._gripper_toggled() else float("nan")
        else:
            gripper = 1.0 if self.gripper_key in keys else -1.0

        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], gripper],
                          dtype=np.float32)
        return TeleopReading(action=action, intervene=bool(intervene),
                             done=bool(done), discard=bool(discard))

    def _gripper_toggled(self) -> bool:
        return self.gripper_key in self._last_trigger
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `python -m pytest tests/test_keyboard_device.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add mile_franka/teleop/keyboard.py tests/test_keyboard_device.py
git commit -m "feat(teleop): KeyboardDevice (tutorial default teleop)"
```

### Task B3: Wire `intervener: keyboard` into train_mile.py

**Files:**
- Modify: `scripts/train_mile.py:262-276`

- [ ] **Step 1: Add the keyboard branch**

After the `elif which == 'joystick':` block (ends at line 274) and before `else:`, insert:

```python
        elif which == 'keyboard':
            from mile_franka.teleop.keyboard import KeyboardDevice
            kb_scale = config['experiment'].get('keyboard_translation_scale', 0.02)
            intervener = TeleopIntervener(KeyboardDevice(translation_scale=kb_scale))
```

- [ ] **Step 2: Verify the selection parses**

Run: `python -c "import ast; ast.parse(open('scripts/train_mile.py').read()); print('syntax ok')"`
Expected: `syntax ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_mile.py
git commit -m "feat(train): select KeyboardDevice via intervener: keyboard"
```

---

## Phase C — The guarded MILE-loss exercise

### Task C1: Create the tutorial package and the reference solution

**Files:**
- Create: `mile_franka/tutorial/__init__.py`
- Create: `mile_franka/tutorial/_loss_solution.py`

- [ ] **Step 1: Create the package marker**

`mile_franka/tutorial/__init__.py`:
```python
"""Tutorial-only scaffolding (loss exercise, loaders). Not part of the MILE method."""
```

- [ ] **Step 2: Create the reference solution (copied verbatim from production)**

`mile_franka/tutorial/_loss_solution.py` — copy the exact body of `mile_cont_loss_fn` from `mile/algorithm.py:116-145` so the answer key matches production behaviour:
```python
"""Reference answer for the loss exercise — identical to mile.algorithm.mile_cont_loss_fn."""
import torch
import torch.distributions as D
import torch.nn.functional as F

from stable_baselines3.common.distributions import sum_independent_dims

device = "cuda" if torch.cuda.is_available() else "cpu"


def _disc_loss(pred_probs, gt_labels, reduction="mean"):
    pred_probs = torch.clamp(pred_probs, 1e-7, 1 - 1e-7)
    return F.nll_loss(torch.log(pred_probs), gt_labels, reduction=reduction)


def mile_cont_loss_fn(intervention_prob, mu, log_std, ground_truth_action,
                      ground_truth_intervention, LAMBDA1=1.0, LAMBDA2=1.0,
                      reduction="mean"):
    discrete_loss = _disc_loss(intervention_prob, ground_truth_intervention, reduction=reduction)
    idx = torch.logical_and(ground_truth_intervention == 1, intervention_prob[:, -1] > 0.0)
    if idx.sum() == 0:
        continuous_loss = torch.tensor(0.0).to(device)
    else:
        dist = D.Normal(mu[idx], log_std[idx].exp())
        log_prob = sum_independent_dims(dist.log_prob(ground_truth_action[idx]))
        continuous_loss = -log_prob.mean()
    loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss
    return loss, continuous_loss, discrete_loss
```

- [ ] **Step 3: Commit**

```bash
git add mile_franka/tutorial/__init__.py mile_franka/tutorial/_loss_solution.py
git commit -m "feat(tutorial): loss-exercise package + reference solution"
```

### Task C2: Write the loss-exercise test (the green-light)

**Files:**
- Create: `tests/test_loss_exercise.py`

- [ ] **Step 1: Write the test**

```python
import numpy as np
import torch
import pytest

from mile_franka.tutorial._loss_solution import mile_cont_loss_fn as REF


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    n, a = 8, 4
    intervention_prob = torch.softmax(torch.randn(n, 2, generator=g), dim=-1)
    mu = torch.randn(n, a, generator=g)
    log_std = torch.zeros(n, a)
    gt_action = torch.randn(n, a, generator=g)
    gt_interv = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    return intervention_prob, mu, log_std, gt_action, gt_interv


def _candidate():
    from mile_franka.tutorial.loss_exercise import mile_cont_loss_fn as C
    return C


def test_solution_matches_itself_known_values():
    b = _batch()
    loss, cont, disc = REF(*b)
    assert torch.isfinite(loss)
    # disc loss = NLL of the 2-class intervention head; cont loss > 0 (some nu=1 steps)
    assert cont.item() > 0
    assert disc.item() > 0


def test_candidate_matches_reference_when_implemented():
    try:
        C = _candidate()
        out = C(*_batch())
    except NotImplementedError:
        pytest.skip("exercise not yet implemented (blank scaffold)")
    ref = REF(*_batch())
    for c, r in zip(out, ref):
        assert torch.allclose(c, r, atol=1e-6), "candidate loss != reference"


def test_continuous_term_only_uses_intervention_steps():
    # If all nu=0, continuous loss must be exactly 0.
    ip, mu, ls, ga, _ = _batch()
    gi = torch.zeros(8, dtype=torch.long)
    _, cont, _ = REF(ip, mu, ls, ga, gi)
    assert cont.item() == 0.0
```

- [ ] **Step 2: Run — reference tests pass, candidate skips (no exercise yet)**

Run: `python -m pytest tests/test_loss_exercise.py -v`
Expected: `test_solution_matches_itself_known_values` PASS, `test_continuous_term_only_uses_intervention_steps` PASS, `test_candidate_matches_reference_when_implemented` FAIL/ERROR (no `loss_exercise` module yet).

### Task C3: Create the blanked exercise scaffold

**Files:**
- Create: `mile_franka/tutorial/loss_exercise.py`

- [ ] **Step 1: Write the blanked scaffold with the docstring spec**

```python
"""MILE loss exercise — IMPLEMENT THIS.

Implement the continuous MILE loss used to train the policy and the mental model.

Inputs (all torch tensors):
- intervention_prob: (batch, 2) — p(nu=0), p(nu=1) from the intervention model.
- mu:                (batch, action_dim) — predicted action mean.
- log_std:           (batch, action_dim) — predicted action log-std.
- ground_truth_action: (batch, action_dim) — the HUMAN action.
- ground_truth_intervention: (batch,) — nu in {0,1} (long).

Return a 3-tuple: (loss, continuous_loss, discrete_loss) where
- discrete_loss = NLL of the intervention head vs ground_truth_intervention
  (clamp probs to [1e-7, 1-1e-7]; use F.nll_loss on log-probs).
- continuous_loss = Gaussian NLL of ground_truth_action under Normal(mu, exp(log_std)),
  averaged, computed ONLY on rows where nu==1 (and intervention_prob[:,-1] > 0).
  If there are no such rows, continuous_loss = tensor(0.0).
- loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss.

Run `make tutorial-check-loss` until it is green. If you run out of time, the answer
key is applied automatically when you train.
"""
import torch
import torch.distributions as D
import torch.nn.functional as F

from stable_baselines3.common.distributions import sum_independent_dims

device = "cuda" if torch.cuda.is_available() else "cpu"


def mile_cont_loss_fn(intervention_prob, mu, log_std, ground_truth_action,
                      ground_truth_intervention, LAMBDA1=1.0, LAMBDA2=1.0,
                      reduction="mean"):
    raise NotImplementedError("TODO: implement the MILE loss (see this file's docstring)")
```

- [ ] **Step 2: Run — candidate test now ERRORS into skip via NotImplementedError**

Run: `python -m pytest tests/test_loss_exercise.py -v`
Expected: `test_candidate_matches_reference_when_implemented` SKIPPED ("exercise not yet implemented"); the two reference tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_loss_exercise.py mile_franka/tutorial/loss_exercise.py
git commit -m "feat(tutorial): blanked MILE-loss exercise + green-light test"
```

### Task C4: Loss loader (validate + auto-apply answer key)

**Files:**
- Create: `mile_franka/tutorial/loss_loader.py`

- [ ] **Step 1: Write the loader**

```python
"""Resolve which loss to train with: the participant's exercise if correct, else the
reference answer key (auto-apply). Used by scripts/tutorial_train.py to monkeypatch
mile.algorithm.mile_cont_loss_fn WITHOUT editing production code.
"""
import torch

from mile_franka.tutorial import _loss_solution as solution


def _sample_batch():
    g = torch.Generator().manual_seed(0)
    ip = torch.softmax(torch.randn(8, 2, generator=g), dim=-1)
    mu = torch.randn(8, 4, generator=g)
    ls = torch.zeros(8, 4)
    ga = torch.randn(8, 4, generator=g)
    gi = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    return ip, mu, ls, ga, gi


def resolve_loss_fn():
    """Return (loss_fn, used_answer_key: bool)."""
    from mile_franka.tutorial import loss_exercise
    b = _sample_batch()
    ref = solution.mile_cont_loss_fn(*b)
    try:
        cand = loss_exercise.mile_cont_loss_fn(*b)
        ok = all(torch.allclose(c, r, atol=1e-6) for c, r in zip(cand, ref))
    except Exception:
        ok = False
    if ok:
        print("[tutorial] using YOUR loss implementation ✔")
        return loss_exercise.mile_cont_loss_fn, False
    print("[tutorial] your loss is missing/incorrect — applying the answer key so you can train.")
    return solution.mile_cont_loss_fn, True
```

- [ ] **Step 2: Test the loader picks the solution when blank**

Run:
```bash
python -c "from mile_franka.tutorial.loss_loader import resolve_loss_fn; fn,ak=resolve_loss_fn(); print('answer_key', ak)"
```
Expected: prints the auto-apply message and `answer_key True` (exercise still blank).

- [ ] **Step 3: Commit**

```bash
git add mile_franka/tutorial/loss_loader.py
git commit -m "feat(tutorial): loss loader with answer-key auto-apply"
```

### Task C5: Tutorial training entrypoint that injects the loss

**Files:**
- Create: `scripts/tutorial_train.py`

**Context:** `InterventionTrainer.__init__` sets `self.loss_fn = mile_cont_loss_fn` by reading the module-level name in `mile/algorithm.py`. Monkeypatching `mile.algorithm.mile_cont_loss_fn` before training makes the trainer use the resolved loss with zero edits to production code.

- [ ] **Step 1: Write the entrypoint**

```python
"""Tutorial training wrapper: inject the participant's loss (or the answer key), then
run the normal train_mile main. Usage: python scripts/tutorial_train.py --config <cfg>
"""
import sys

import mile.algorithm as algorithm
from mile_franka.tutorial.loss_loader import resolve_loss_fn


def main():
    loss_fn, _ = resolve_loss_fn()
    algorithm.mile_cont_loss_fn = loss_fn  # monkeypatch BEFORE trainer construction
    import train_mile  # scripts/ is on sys.path when run from scripts/
    train_mile.main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm train_mile exposes a callable main()**

Run: `grep -n "def main" scripts/train_mile.py`
Expected: a `def main(` exists. If train_mile instead runs under `if __name__ == "__main__":` with inline code, refactor that block into `def main():` and call it from the guard (small, behaviour-preserving edit), then re-run this grep.

- [ ] **Step 3: Syntax check**

Run: `python -c "import ast; ast.parse(open('scripts/tutorial_train.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/tutorial_train.py scripts/train_mile.py
git commit -m "feat(tutorial): training entrypoint that injects the loss exercise"
```

---

## Phase D — Reduced tutorial configs

### Task D1: MetaWorld tutorial config (Tier 1)

**Files:**
- Create: `config_tutorial_metaworld.json`

**Context:** Based on `config.json` but sized to finish in ~10 min and show improvement; `auto_eval` lives under `experiment.rollout`. Uses the downloaded expert paths verified in the spec (§4.1).

- [ ] **Step 1: Write the config**

```json
{
    "train": {"num_epochs": 50, "batch_size": 1024, "lr": 0.0001, "lambda1": 1.0, "lambda2": 1.0},
    "experiment": {
        "name": "tutorial_metaworld",
        "mode": "iterative",
        "validate": {"enabled": false, "every_n_epochs": 100},
        "logging": {"terminal_output_to_txt": false, "log_tb": false, "log_wandb": false},
        "save": {"enabled": true, "every_n_epochs": 0, "on_best_validation": false,
                  "on_best_rollout_return": false, "on_best_rollout_success_rate": true,
                  "outdir": "output_dir"},
        "rollout": {"enabled": true, "n_episodes": 10, "every_n_epochs": 50, "auto_eval": true},
        "env_name": "peg-insert-side-v2",
        "collector": "synthetic",
        "policy_type": "sac",
        "policy_path": "./trained_models/initial_policy",
        "mental_model_type": "bc",
        "gt_mental_model_path": "./trained_models/gt_mental_model",
        "intervention_policy_path": "./trained_models/expert_policy",
        "use_warm_start": true,
        "warm_start_path": "./trained_models/warm_started_mental_model",
        "num_rounds": 2,
        "episodes_per_round": 1,
        "include_offline_dataset": false
    }
}
```

- [ ] **Step 2: Validate it parses and references existing keys**

Run: `python -c "import json; c=json.load(open('config_tutorial_metaworld.json')); assert c['experiment']['rollout']['auto_eval'] is True; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add config_tutorial_metaworld.json
git commit -m "feat(tutorial): reduced MetaWorld Tier-1 config"
```

### Task D2: Franka tutorial config (Tier 3)

**Files:**
- Create: `config_tutorial_franka.json`

- [ ] **Step 1: Write the config** (mirror `config_franka.json`, reduced, keyboard intervener, auto_eval off)

```json
{
    "train": {"num_epochs": 100, "batch_size": 256, "lr": 0.0001, "lambda1": 1.0, "lambda2": 1.0},
    "experiment": {
        "name": "tutorial_franka",
        "mode": "iterative",
        "validate": {"enabled": false, "every_n_epochs": 100},
        "logging": {"terminal_output_to_txt": false, "log_tb": false, "log_wandb": false},
        "save": {"enabled": true, "every_n_epochs": 0, "on_best_validation": false,
                  "on_best_rollout_return": false, "on_best_rollout_success_rate": false,
                  "outdir": "output_dir"},
        "rollout": {"enabled": false, "n_episodes": 0, "every_n_epochs": 0, "auto_eval": false},
        "env_name": "Franka-Stack-Sim-v0",
        "collector": "real",
        "intervener": "keyboard",
        "keyboard_translation_scale": 0.02,
        "policy_type": "bc",
        "policy_path": "./trained_models/franka/base_policy",
        "mental_model_type": "bc",
        "num_rounds": 1,
        "episodes_per_round": 2,
        "include_offline_dataset": false
    }
}
```

- [ ] **Step 2: Validate parse + safety invariant (auto_eval false)**

Run: `python -c "import json; c=json.load(open('config_tutorial_franka.json')); assert c['experiment']['rollout']['auto_eval'] is False and c['experiment']['intervener']=='keyboard'; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add config_tutorial_franka.json
git commit -m "feat(tutorial): reduced Franka Tier-3 config (keyboard, auto_eval off)"
```

---

## Phase E — Artifact bundle (Drive-free distribution)

### Task E1: Bundle assembly script (run once by the instructor)

**Files:**
- Create: `scripts/make_tutorial_artifacts.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Run ONCE by the instructor. Pulls the MetaWorld expert from Drive, adds the Franka
# base policy, bundles into tutorial-artifacts.tar, prints + writes the SHA256.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK=$(mktemp -d)
echo "[1/4] downloading MetaWorld expert from Drive (one-time)…"
python3 -m gdown 1bzKGyOmX1ZCmAWnZiq_sAFRxi3AXvm4t -O "$WORK/trained_models.zip"
( cd "$WORK" && unzip -q trained_models.zip )   # -> $WORK/trained_models/{4 models}

echo "[2/4] adding the Franka base policy…"
mkdir -p "$WORK/trained_models/franka"
cp trained_models/franka/base_policy "$WORK/trained_models/franka/base_policy"

echo "[3/4] taring the bundle…"
tar -C "$WORK" -cf tutorial-artifacts.tar trained_models

echo "[4/4] checksum…"
sha256sum tutorial-artifacts.tar | tee tutorial-artifacts.sha256
echo "Done. Upload tutorial-artifacts.tar as a GitHub Release asset, commit tutorial-artifacts.sha256."
rm -rf "$WORK"
```

- [ ] **Step 2: Make it executable + syntax check**

Run: `chmod +x scripts/make_tutorial_artifacts.sh && bash -n scripts/make_tutorial_artifacts.sh && echo ok`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_tutorial_artifacts.sh
git commit -m "feat(tutorial): one-time artifact bundle assembly script"
```

### Task E2: Fetch script (participant homework, Drive-free)

**Files:**
- Create: `scripts/fetch_artifacts.py`
- Create: `tutorial-artifacts.sha256` (placeholder until E1 is run for real)

- [ ] **Step 1: Create the checksum file**

Until `make_tutorial_artifacts.sh` is run on real data, write a sentinel so the verifier has a target. After the real run (E1), this file is overwritten with the true hash and committed.
```
PENDING-run-scripts/make_tutorial_artifacts.sh  tutorial-artifacts.tar
```

- [ ] **Step 2: Write the fetch script**

```python
"""Download + verify the tutorial artifacts bundle from the GitHub Release (NOT Drive).

Homework step. Set ARTIFACTS_URL (defaults to the repo's latest-release asset) and run
`make fetch-artifacts`. Verifies SHA256 against tutorial-artifacts.sha256, then unpacks
trained_models/ into the repo root.
"""
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAR = ROOT / "tutorial-artifacts.tar"
SUMS = ROOT / "tutorial-artifacts.sha256"
DEFAULT_URL = os.environ.get(
    "ARTIFACTS_URL",
    "https://github.com/rayray2002/mile-franka-tutorial/releases/latest/download/tutorial-artifacts.tar",
)


def expected_sha():
    line = SUMS.read_text().split()[0]
    if line.startswith("PENDING"):
        sys.exit("tutorial-artifacts.sha256 not finalized — instructor must run "
                 "scripts/make_tutorial_artifacts.sh and commit the checksum.")
    return line


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not TAR.exists():
        print(f"downloading {DEFAULT_URL}")
        subprocess.check_call(["curl", "-fL", "-o", str(TAR), DEFAULT_URL])
    got, want = sha256(TAR), expected_sha()
    if got != want:
        sys.exit(f"checksum mismatch: got {got}, want {want}")
    print("checksum ok; unpacking…")
    with tarfile.open(TAR) as t:
        t.extractall(ROOT)
    print("artifacts ready under trained_models/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax check**

Run: `python -c "import ast; ast.parse(open('scripts/fetch_artifacts.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_artifacts.py tutorial-artifacts.sha256
git commit -m "feat(tutorial): Drive-free artifact fetch + checksum verify"
```

---

## Phase F — tutorial-check and make verbs

### Task F1: Environment readiness check

**Files:**
- Create: `scripts/tutorial_check.py`

- [ ] **Step 1: Write the check**

```python
"""Tutorial readiness check (run at home and at 0:05). Asserts imports + artifacts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = [
    "trained_models/initial_policy", "trained_models/expert_policy",
    "trained_models/gt_mental_model", "trained_models/warm_started_mental_model",
    "trained_models/franka/base_policy",
]


def main():
    ok = True
    for mod in ("metaworld", "mile", "mile_franka", "torch", "gymnasium"):
        try:
            __import__(mod)
            print(f"  import {mod}: OK")
        except Exception as e:
            ok = False
            print(f"  import {mod}: FAIL ({e})")
    for rel in REQUIRED:
        p = ROOT / rel
        print(f"  artifact {rel}: {'OK' if p.exists() else 'MISSING'}")
        ok = ok and p.exists()
    if not ok:
        sys.exit("tutorial-check FAILED — see above (run `make fetch-artifacts` for artifacts).")
    print("tutorial-check OK — you're ready.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (expect MISSING artifacts until fetched)**

Run: `python scripts/tutorial_check.py || true`
Expected: import lines OK (in-container), artifact lines MISSING until `fetch-artifacts` is run; exit nonzero. This confirms detection works.

- [ ] **Step 3: Commit**

```bash
git add scripts/tutorial_check.py
git commit -m "feat(tutorial): environment readiness check"
```

### Task F2: Add the tutorial-* make verbs

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Append the verbs** (follow the existing `## help` comment style; assume the in-container exec pattern used by `mile`/`eval-base` — reuse the same `docker compose … exec sim` prefix variable the Makefile already defines; match the existing target that runs `train_mile.py` for the exact invocation form)

```makefile
fetch-artifacts:             ## download + verify the tutorial artifacts bundle (homework; not Drive)
	python3 scripts/fetch_artifacts.py

tutorial-check:              ## assert imports + artifacts are present (run at home and at 0:05)
	python3 scripts/tutorial_check.py

tutorial-check-loss:         ## green-light test for the MILE-loss exercise
	python3 -m pytest tests/test_loss_exercise.py -v

tutorial-metaworld:          ## Tier 1: run the MetaWorld synthetic loop (uses your loss)
	cd scripts && python3 tutorial_train.py --config ../config_tutorial_metaworld.json

tutorial-collect-train:      ## Tier 3: collect interventions (keyboard) + train (uses your loss)
	cd scripts && python3 tutorial_train.py --config ../config_tutorial_franka.json

tutorial-fake:               ## Tier 2: run the mediocre base policy on the fake backend
	python3 scripts/smoke_franka_env.py
```

(`eval-base`/`eval-after`: reuse the existing `eval-base`; document `eval-after` as `make eval-base` re-run after training, or add an alias target pointing at the MILE-trained policy via the existing `eval-mile`.)

- [ ] **Step 2: Verify the verbs are listed**

Run: `make help 2>/dev/null | grep -E "tutorial-|fetch-artifacts" || grep -nE "^tutorial-|^fetch-artifacts" Makefile`
Expected: the new targets appear.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat(tutorial): make verbs (check, loss, metaworld, collect-train, fake)"
```

---

## Phase G — Documentation set

Each doc is a concrete, copy-pasteable file under `docs/tutorial/`. Content requirements are listed per file; write real prose + real commands (no "TBD"). Verify each by running every command it contains.

### Task G1: Pre-session setup guide

**Files:**
- Create: `docs/tutorial/00-setup.md`

- [ ] **Step 1: Write it.** Must contain, as numbered steps with exact commands: host prereqs (Ubuntu 22+, NVIDIA driver, `nvidia-container-toolkit`, Docker + compose plugin, 50–100 GB, ethernet); `git clone` of the repo and the hucebot base build (`git clone https://github.com/hucebot/multipanda_ros2 && cd multipanda_ros2 && docker compose build`); `make build`; `make up`; `make fetch-artifacts`; `make tutorial-check` (showing the expected `tutorial-check OK` output); pairing + 5090-tower instructions for weak laptops; a "you're ready when you see `tutorial-check OK`" line.
- [ ] **Step 2: Verify commands** — run each command block on the host where possible; confirm `make tutorial-check` output text matches what the doc claims.
- [ ] **Step 3: Commit** `git add docs/tutorial/00-setup.md && git commit -m "docs(tutorial): pre-session setup guide"`

### Task G2: Concepts primer

**Files:**
- Create: `docs/tutorial/01-concepts.md`

- [ ] **Step 1: Write it.** 5-minute primer: what intervention learning is; ν (the intervention flag); the intervention model; joint policy + mental-model training; the BCE + Gaussian-NLL loss in plain terms; a table mapping each tier (1–4) to the method. Link to `02-loss-exercise.md`.
- [ ] **Step 2: Self-review** for accuracy against `mile/computational_model.py` and `mile/algorithm.py` (names/claims correct).
- [ ] **Step 3: Commit** `git add docs/tutorial/01-concepts.md && git commit -m "docs(tutorial): MILE concepts primer"`

### Task G3: Loss-exercise handout

**Files:**
- Create: `docs/tutorial/02-loss-exercise.md`

- [ ] **Step 1: Write it.** The math (BCE on ν; Gaussian NLL of the human action on ν=1 steps; `loss = λ1·cont + λ2·disc`); the exact file to edit (`mile_franka/tutorial/loss_exercise.py`); the input tensor shapes (from the scaffold docstring); how to run `make tutorial-check-loss` and read green/red; the auto-apply behaviour at train time; a worked explanation to read *after* attempting.
- [ ] **Step 2: Verify** — run `make tutorial-check-loss` and confirm the doc's described output matches.
- [ ] **Step 3: Commit** `git add docs/tutorial/02-loss-exercise.md && git commit -m "docs(tutorial): loss-exercise handout"`

### Task G4: Per-tier walkthrough

**Files:**
- Create: `docs/tutorial/03-tier-walkthrough.md`

- [ ] **Step 1: Write it.** The 0:00→2:00 flow from spec §7 as numbered steps with exact `make` verbs, expected logs/numbers, and "what you should see" callouts per tier (incl. reading `eval-base`/`eval-after`, the Tier-1 success-rate climb, the Tier-3 keyboard collect loop).
- [ ] **Step 2: Verify** — dry-run each `make` verb name against `make help`; confirm every referenced verb exists.
- [ ] **Step 3: Commit** `git add docs/tutorial/03-tier-walkthrough.md && git commit -m "docs(tutorial): per-tier walkthrough"`

### Task G5: Teleop reference

**Files:**
- Create: `docs/tutorial/04-teleop.md`

- [ ] **Step 1: Write it.** Keyboard map table (w/s/a/d/q/e, space=clutch/ν, g=gripper, enter=done, backspace=discard) matching `mile_franka/teleop/keyboard.py` defaults; the segment-toggle intervention model explained; the gamepad (`intervener: joystick`) and Vive (`intervener: vive`) alternatives.
- [ ] **Step 2: Verify** the key map exactly matches the `KeyboardDevice` defaults (cross-check the source).
- [ ] **Step 3: Commit** `git add docs/tutorial/04-teleop.md && git commit -m "docs(tutorial): teleop reference"`

### Task G6: Troubleshooting / FAQ

**Files:**
- Create: `docs/tutorial/05-troubleshooting.md`

- [ ] **Step 1: Write it.** Each known failure mode + one-line fix: GL/rendering for `sim-gui` (software GL note), X11 (`xhost +local:root`), GPU not visible (`gpus: all` / nvidia runtime), artifact checksum mismatch (`make fetch-artifacts` again / use USB), sim not up before training (`make sim-up` in another terminal), keyboard window not focused (click the teleop window), `hucebot:franka-humble` pull-denied (build the base locally).
- [ ] **Step 2: Commit** `git add docs/tutorial/05-troubleshooting.md && git commit -m "docs(tutorial): troubleshooting/FAQ"`

### Task G7: Instructor runbook

**Files:**
- Create: `docs/tutorial/06-instructor-runbook.md`

- [ ] **Step 1: Write it.** France-day pre-session prep (spec §8: `make calibrate-camera`, start hucebot controller, re-tune `COST_LOOKUP`, optional Vive wiring); the real-FR3 station queue + safety invariants (from the status doc); timing cuts if running long (drop Tier 2, shrink rounds/epochs); the `intervener: keyboard → vive` switch; per-tier talking points.
- [ ] **Step 2: Commit** `git add docs/tutorial/06-instructor-runbook.md && git commit -m "docs(tutorial): instructor runbook"`

### Task G8: Point the README at the tutorial docs

**Files:**
- Modify: `README.md` (the "MILE on Franka — block-stacking tutorial" section intro)

- [ ] **Step 1: Add a pointer** near the top of the Franka section: `> **Summer-school participants:** start at [docs/tutorial/00-setup.md](docs/tutorial/00-setup.md).`
- [ ] **Step 2: Commit** `git add README.md && git commit -m "docs: link README to the tutorial docs"`

---

## Self-review notes (addressed)

- **Spec coverage:** §4 image → A1/A2/A4; §4 twin → A3; §4.1 artifacts → E1/E2; §5 keyboard → B1–B3; §6 loss TODO → C1–C5; §7 flow → make verbs F2 + walkthrough G4; §8 France prep → G7; §9 build items → all phases; §13 docs → G1–G8. Deferred §14 items intentionally absent.
- **Production code untouched:** the loss is injected by monkeypatch in `scripts/tutorial_train.py`; `mile/algorithm.py` is not edited (only `scripts/train_mile.py` gains a keyboard branch + optional `main()` extraction).
- **Type consistency:** `KeyboardDevice` exposes `read()`, `reset()`, `sync_gripper_state()` matching `JoystickDevice`/`TeleopIntervener`; the exercise + solution + loader all use the `mile_cont_loss_fn(intervention_prob, mu, log_std, ground_truth_action, ground_truth_intervention, LAMBDA1, LAMBDA2, reduction)` signature from `mile/algorithm.py`.
- **Open verification dependencies:** A4 needs the `hucebot:franka-humble` base; D/F live runs need `make sim-up`; E2's real checksum is finalized only after E1 runs on real data (Step F-level checks tolerate the PENDING sentinel).
