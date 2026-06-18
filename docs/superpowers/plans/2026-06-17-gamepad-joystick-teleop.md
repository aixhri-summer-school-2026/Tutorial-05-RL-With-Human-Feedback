# Gamepad Joystick Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Xbox/PS gamepad as a third real-human teleop device on the phase-a sim MILE loop, alongside the existing SpaceMouse, with no change to the abstract teleop interfaces, the collector, or the Box dataset.

**Architecture:** A new `JoystickDevice(TeleopDevice)` mirrors `SpaceMouseDevice` exactly — an injectable raw reader (so the mapping logic loads/tests with no hardware) wrapping `pygame.joystick`. It emits the same `[dx, dy, dz, gripper]` 4-DoF `TeleopReading`. ν is True on **clutch-hold OR stick-motion OR gripper-button**. It is selected via `intervener: joystick` in `config_franka.json`, wired through one new branch in `train_mile.py`. A `scripts/joystick_check.py` (with a hardware-free `--fake` mode) plus `make joystick-check` and a commented compose `/dev/input` passthrough complete the device path.

**Tech Stack:** Python 3.10, `pygame==2.6.1` (already in `docker/requirements-mile.lock.txt` — no new dependency), numpy, the existing `mile_franka.teleop` package, Docker compose `sim` service.

**Convention reminders:** This repo has **no pytest suite**; smoke scripts run in-container are the de-facto tests. Verification commands use the `make`/`docker compose exec sim` path because pygame/numpy live in the container, not on the host. No Claude/AI attribution in commits (CLAUDE.md). Reference design: `docs/superpowers/specs/2026-06-17-phase-a-sim-spacemouse-design.md` §5.6.

---

### Task 1: `JoystickDevice` teleop class

**Files:**
- Create: `mile_franka/teleop/joystick.py`
- Reference (read, do not modify): `mile_franka/teleop/spacemouse.py`, `mile_franka/teleop/base.py`

- [ ] **Step 1: Write `mile_franka/teleop/joystick.py`**

The reader contract: a zero-arg callable returning a snapshot object exposing two sequences,
`.axes` and `.buttons`. The default reader lazily imports pygame, pumps its event queue, and
reads joystick 0. Indices are constructor args (pads vary); defaults target a typical Xbox layout
(left stick = axes 0/1, right stick vertical = axis 4; A button = 0, RB = 5, Start = 7).

```python
"""Gamepad (Xbox/PS) teleop device — the device the original MILE paper used.

Mirrors SpaceMouseDevice: the raw reader is injectable so the mapping logic runs without
hardware. The default reader lazily imports pygame (already in the image), so the module
loads on machines without a controller.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from mile_franka.teleop.base import TeleopDevice, TeleopReading


class JoystickDevice(TeleopDevice):
    """Maps a gamepad to a 4-DoF [dx, dy, dz, gripper] action with clutch semantics.

    ν (intervene) is True when the clutch is held OR any mapped axis exceeds the deadband
    OR the gripper button is held — a superset of SpaceMouse's `moved or gripper_close`.

    Args:
        translation_scale: meters of EE delta per unit of stick deflection.
        deadband: stick deflection below this (abs) is treated as zero.
        ax_x, ax_y, ax_z: axis indices mapped to dx, dy, dz.
        clutch_button, gripper_button, done_button: button indices.
        reader: zero-arg callable returning an object with `.axes` and `.buttons` sequences.
                Defaults to a pygame-backed reader.
    """

    def __init__(self, translation_scale: float = 0.02, deadband: float = 0.1,
                 ax_x: int = 0, ax_y: int = 1, ax_z: int = 4,
                 clutch_button: int = 5, gripper_button: int = 0, done_button: int = 7,
                 reader: Optional[Callable[[], object]] = None):
        self.translation_scale = translation_scale
        self.deadband = deadband
        self.ax_x, self.ax_y, self.ax_z = ax_x, ax_y, ax_z
        self.clutch_button = clutch_button
        self.gripper_button = gripper_button
        self.done_button = done_button
        self._reader = reader if reader is not None else self._default_reader()

    @staticmethod
    def _default_reader() -> Callable[[], object]:
        import pygame  # lazy: only needed with real hardware

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise SystemExit("No joystick found — check /dev/input mapping and that a pad is connected")
        js = pygame.joystick.Joystick(0)
        js.init()

        class _Snapshot:
            __slots__ = ("axes", "buttons")

        def _read() -> object:
            pygame.event.pump()
            snap = _Snapshot()
            snap.axes = [js.get_axis(i) for i in range(js.get_numaxes())]
            snap.buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
            return snap

        return _read

    @staticmethod
    def _axis(axes: Sequence[float], idx: int) -> float:
        return float(axes[idx]) if 0 <= idx < len(axes) else 0.0

    @staticmethod
    def _button(buttons: Sequence[int], idx: int) -> bool:
        return bool(buttons[idx]) if 0 <= idx < len(buttons) else False

    def read(self) -> TeleopReading:
        s = self._reader()
        axes = getattr(s, "axes", ())
        buttons = getattr(s, "buttons", ())

        trans = np.array([self._axis(axes, self.ax_x),
                          self._axis(axes, self.ax_y),
                          self._axis(axes, self.ax_z)], dtype=np.float32)
        trans[np.abs(trans) < self.deadband] = 0.0
        action_xyz = trans * self.translation_scale

        clutch = self._button(buttons, self.clutch_button)
        gripper_close = self._button(buttons, self.gripper_button)
        done = self._button(buttons, self.done_button)

        gripper = 1.0 if gripper_close else -1.0
        moved = bool(np.any(action_xyz != 0.0))
        intervene = clutch or moved or gripper_close  # ν = clutch OR motion OR gripper

        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], gripper],
                          dtype=np.float32)
        return TeleopReading(action=action, intervene=intervene, done=done)

    def close(self) -> None:
        try:
            import pygame

            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
```

- [ ] **Step 2: Verify the mapping logic with a hardware-free injected reader (in-container)**

Run:

```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && python3 - <<PY
import numpy as np
from mile_franka.teleop.joystick import JoystickDevice

class Snap:
    def __init__(self, axes, buttons):
        self.axes, self.buttons = axes, buttons

# neutral: no clutch, no motion, no buttons -> nu=0, gripper open
dev = JoystickDevice(reader=lambda: Snap([0.0]*8, [0]*8))
r = dev.read()
assert r.intervene is False, r
assert r.action[3] == -1.0, r        # gripper open
assert np.allclose(r.action[:3], 0), r

# clutch held (button 5), sticks neutral -> nu=1 via clutch alone
dev = JoystickDevice(reader=lambda: Snap([0.0]*8, [0,0,0,0,0,1,0,0]))
assert dev.read().intervene is True

# left stick pushed past deadband, no clutch -> nu=1 via motion; scaled delta
dev = JoystickDevice(reader=lambda: Snap([0.5,0.0,0.0,0.0,0.0,0,0,0], [0]*8))
r = dev.read()
assert r.intervene is True
assert abs(r.action[0] - 0.5*0.02) < 1e-6, r

# tiny stick below deadband -> zeroed, nu=0
dev = JoystickDevice(reader=lambda: Snap([0.05,0.0,0.0,0.0,0.0,0,0,0], [0]*8))
r = dev.read()
assert r.intervene is False and r.action[0] == 0.0, r

# gripper button (0) -> close (+1) and nu=1; done button (7) -> done
dev = JoystickDevice(reader=lambda: Snap([0.0]*8, [1,0,0,0,0,0,0,1]))
r = dev.read()
assert r.action[3] == 1.0 and r.intervene is True and r.done is True, r

print("OK joystick mapping")
PY'
```

Expected: prints `OK joystick mapping` with no assertion errors.

- [ ] **Step 3: Commit**

```bash
git add mile_franka/teleop/joystick.py
git commit -m "teleop: add gamepad JoystickDevice (clutch-or-motion intervene)"
```

---

### Task 2: `joystick_check` sanity script + Makefile verb

**Files:**
- Create: `scripts/joystick_check.py`
- Modify: `Makefile` (`.PHONY` line 11; add a target near `spacemouse-check` at line 57-58)
- Reference (read, do not modify): `scripts/spacemouse_check.py`

- [ ] **Step 1: Write `scripts/joystick_check.py`**

Real mode prints live axes/buttons; `--fake` mode exercises the read path with a synthetic
reader so the script verifies without a controller attached.

```python
#!/usr/bin/env python3
"""Gamepad sanity check: print live axes/buttons. Blocks until Ctrl-C.

The whole gamepad path is blocked until this prints nonzero values when the sticks are
pushed. Run inside the container via `make joystick-check`. Use `--fake` to exercise the
read path with no controller attached.
"""
import argparse
import time

from mile_franka.teleop.joystick import JoystickDevice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true",
                        help="use a synthetic reader (no hardware) to smoke-test the read path")
    args = parser.parse_args()

    if args.fake:
        class Snap:
            axes = [0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            buttons = [1, 0, 0, 0, 0, 1, 0, 0]
        dev = JoystickDevice(reader=lambda: Snap())
        r = dev.read()
        print(f"[fake] action={r.action} intervene={r.intervene} done={r.done}")
        return

    dev = JoystickDevice()
    print("Joystick open. Push the sticks / press buttons; Ctrl-C to stop.")
    try:
        while True:
            r = dev.read()
            print(f"action={r.action} intervene={r.intervene} done={r.done}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the Makefile target**

In `Makefile`, append `joystick-check` to the `.PHONY` list on line 11 (after `spacemouse-check`):

```make
.PHONY: build up down shell sim-up sim-gui collect-mediocre collect-expert base-policy mile spacemouse-check joystick-check eval-base
```

And add this target immediately after the existing `spacemouse-check` target (after line 58):

```make
joystick-check:              ## print live gamepad axes/buttons (sanity check; Ctrl-C to stop)
	$(call RUN,python3 scripts/joystick_check.py)
```

- [ ] **Step 3: Verify the fake path runs in-container**

Run:

```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && python3 scripts/joystick_check.py --fake'
```

Expected: one line like `[fake] action=[ 0.01 -0.005 0. 1. ] intervene=True done=False` (gripper +1 because button 0 is set, intervene True via clutch button 5).

- [ ] **Step 4: Verify the Makefile verb is wired**

Run:

```bash
make -n joystick-check
```

Expected: prints a `docker compose ... exec sim ... python3 scripts/joystick_check.py` command (dry-run, no error).

- [ ] **Step 5: Commit**

```bash
git add scripts/joystick_check.py Makefile
git commit -m "make: add joystick-check sanity verb (with --fake smoke mode)"
```

---

### Task 3: Compose `/dev/input` passthrough (commented, config-only)

**Files:**
- Modify: `docker/docker-compose.yml` (the `devices:` block, ~line 28-30)

- [ ] **Step 1: Add the commented gamepad device mapping**

In `docker/docker-compose.yml`, under `devices:`, directly below the SpaceMouse `hidraw0` line, add the commented gamepad entries (keep them commented so it stays config-only, not a rebuild — mirroring the existing camera line):

```yaml
    devices:
      - /dev/hidraw0:/dev/hidraw0          # SpaceMouse (3Dconnexion) — verify node with: lsusb | grep -i 3dconnex; ls -l /dev/hidraw*
    #   - /dev/input/js0:/dev/input/js0    # gamepad (Xbox/PS) — verify node with: ls -l /dev/input/js*; ls -l /dev/input/event*
    #   - /dev/input/event0:/dev/input/event0  # gamepad SDL evdev node — pick the controller's event* (cat /proc/bus/input/devices)
    #   - /dev/video0:/dev/video0          # camera (France) — uncomment on hardware
```

- [ ] **Step 2: Verify compose still parses**

Run:

```bash
docker compose -f docker/docker-compose.yml config >/dev/null && echo "compose OK"
```

Expected: prints `compose OK` (the commented lines must not break YAML parsing).

- [ ] **Step 3: Commit**

```bash
git add docker/docker-compose.yml
git commit -m "compose: add commented gamepad /dev/input passthrough"
```

---

### Task 4: Wire `intervener: joystick` into `train_mile.py`

**Files:**
- Modify: `scripts/train_mile.py:248-252` (the intervener selection branch)
- Reference (read): `config_franka.json:13`

- [ ] **Step 1: Add the joystick branch**

In `scripts/train_mile.py`, locate the intervener selection (around lines 248-252):

```python
        elif which == 'spacemouse':
            from mile_franka.teleop.spacemouse import SpaceMouseDevice
            intervener = TeleopIntervener(SpaceMouseDevice())
        else:
            raise ValueError(f'Unknown intervener: {which}')
```

Insert a `joystick` branch before the `else`:

```python
        elif which == 'spacemouse':
            from mile_franka.teleop.spacemouse import SpaceMouseDevice
            intervener = TeleopIntervener(SpaceMouseDevice())
        elif which == 'joystick':
            from mile_franka.teleop.joystick import JoystickDevice
            intervener = TeleopIntervener(JoystickDevice())
        else:
            raise ValueError(f'Unknown intervener: {which}')
```

- [ ] **Step 2: Verify the branch selects the device without instantiating hardware**

This confirms the wiring picks `JoystickDevice` for `which == 'joystick'`. We inject a fake reader
so no controller is needed (the real branch calls `JoystickDevice()` which would open hardware — the
test below targets the import + adapter wiring path directly).

Run:

```bash
docker compose -f docker/docker-compose.yml exec sim bash -lc 'source scripts/in_container_env.sh && python3 - <<PY
from mile_franka.collect import TeleopIntervener
from mile_franka.teleop.joystick import JoystickDevice

class Snap:
    axes = [0.0]*8
    buttons = [0,0,0,0,0,1,0,0]   # clutch held
intervener = TeleopIntervener(JoystickDevice(reader=lambda: Snap()))
import numpy as np
action, intervene, done = intervener.intervene(np.zeros(72, dtype=np.float32), np.zeros(4, dtype=np.float32))
assert intervene is True, (action, intervene, done)
print("OK joystick intervener wiring")
PY'
```

Expected: prints `OK joystick intervener wiring`.

- [ ] **Step 3: Confirm an unknown intervener still raises (regression guard)**

Run:

```bash
grep -n "Unknown intervener" scripts/train_mile.py
```

Expected: the `raise ValueError(f'Unknown intervener: {which}')` line is still present (the `joystick` branch was inserted *before* `else`, not replacing it).

- [ ] **Step 4: Commit**

```bash
git add scripts/train_mile.py
git commit -m "train_mile: wire intervener: joystick to JoystickDevice"
```

---

### Task 5: Documentation touch-ups

**Files:**
- Modify: `CLAUDE.md` (the `mile_franka/teleop/` bullet under "The `mile_franka` package")
- Modify: `config_franka.json:13` — leave value as `spacemouse`; add no code, just confirm the option is documented

- [ ] **Step 1: Update the teleop bullet in CLAUDE.md**

In `CLAUDE.md`, find the `mile_franka/teleop/` bullet:

```
- **`mile_franka/teleop/`** — `TeleopDevice`/`TeleopReading` ABCs (`base.py`);
  `SpaceMouseDevice` (`spacemouse.py`, injectable raw reader so it loads without hardware).
```

Replace it with:

```
- **`mile_franka/teleop/`** — `TeleopDevice`/`TeleopReading` ABCs (`base.py`);
  `SpaceMouseDevice` (`spacemouse.py`) and `JoystickDevice` (`joystick.py`, pygame gamepad,
  ν = clutch-or-motion) — both use an injectable raw reader so they load/test without hardware.
  Select via `intervener: spacemouse|joystick` in `config_franka.json`.
```

- [ ] **Step 2: Add the `make joystick-check` line to the Commands block in CLAUDE.md**

In `CLAUDE.md`, the Franka commands block lists `make` verbs. After the `make shell` line, add:

```
make joystick-check           # print live gamepad axes/buttons (sanity check)
```

(If a `make spacemouse-check` line is absent there too, that is acceptable — only add the joystick line.)

- [ ] **Step 3: Verify the docs reference resolves**

Run:

```bash
grep -n "JoystickDevice\|joystick-check\|intervener: spacemouse|joystick" CLAUDE.md
```

Expected: prints the two added references.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document JoystickDevice and intervener: joystick option"
```

---

## Self-Review

**Spec coverage (§5.6 of the phase-a design):**
- New `mile_franka/teleop/joystick.py` `JoystickDevice(TeleopDevice)`, injectable reader, lazy pygame, Xbox-default indices, `translation_scale`/`deadband` → **Task 1**.
- ν = clutch OR motion OR gripper, action zeroed when no signal → **Task 1, Step 1 + verified Step 2**.
- Same `[dx, dy, dz, gripper]` / `TeleopReading` output → **Task 1** (asserted in Step 2).
- `train_mile.py` `elif which == 'joystick'` branch → **Task 4**.
- `config_franka.json` stays `spacemouse`, one-word switch → **Task 5, Step 2** (left unchanged, documented).
- Commented `/dev/input` compose passthrough → **Task 3**.
- `make joystick-check` mirroring `make spacemouse-check` → **Task 2**.
- No new dependency (pygame already in lock file) → no requirements change in any task. ✓
- Forward-compat / device-agnostic contract untouched → no edit to `base.py` or `collect.py`. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every verify step shows a command and expected output. ✓

**Type consistency:** Reader contract (`.axes`, `.buttons` sequences) is identical across `JoystickDevice._default_reader`, the Task 1/2/4 fake `Snap` objects, and `joystick_check.py`. `TeleopReading(action, intervene, done)` matches `base.py`. `TeleopIntervener.intervene(state, rollout_action) -> (action, intervene, done)` matches `collect.py`. Button/axis index defaults (clutch 5, gripper 0, done 7, ax 0/1/4) are consistent between the class defaults and every test fixture. ✓
