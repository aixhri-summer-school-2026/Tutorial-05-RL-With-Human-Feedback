# Phase (a): Real Human-in-the-Loop with SpaceMouse / Gamepad on the Sim Robot

**Date:** 2026-06-17
**Author:** rayray2002
**Status:** Design (for review)

First phase of the agreed roadmap **(a) sim + SpaceMouse → (b) real FR3 + SpaceMouse + AprilTag
→ (c) France = Vive**. Parent design: `2026-06-15-mile-franka-stacking-design.md`; builds on the
docker increment `2026-06-16-mile-docker-image-and-sim-demos-design.md`. Status/TODOs:
`docs/superpowers/notes/2026-06-17-status-and-todos.md`.

## 1. Goal

Put a **real human with a real SpaceMouse** in the MILE intervention loop on the **sim** Panda,
and show the loop works *as intended*: a mediocre base policy, a human who takes over when it is
about to fail, and a policy that **measurably improves across N=2–3 rounds**. This is the cheapest
place to validate the research claim and to calibrate the intervention cost before touching the
real FR3. SpaceMouse mirrors Vive's clutch + delta semantics, so it doubles as a France rehearsal.

This is also acceptance gate 5 of the parent spec (teleop intervention collection), done for real.

## 2. Why this phase matters (not just plumbing)

All the plumbing can be green and the policy still might not improve — that is the actual risk.
Two things can only be discovered with a real human in the loop:
- **Does intervention data actually improve the policy** on this task/obs space?
- **`COST_LOOKUP` calibration:** `[cost, cdf_scale]` governs *when the model believes a human would
  intervene*. It must be tuned against a **real human's** intervention rate, which requires real
  teleop data — so this phase is where it gets calibrated, before France.

## 3. Prerequisites (carry-over from the docker increment review)

These must be true before the SpaceMouse work is meaningful (see status/TODOs doc):
1. **Commit the env layer** (`ros_backend.py`, `franka_env.py`, `registration.py`, `mujoco_gt.py`,
   `train_mile.py`) so the coherence fixes are durable.
2. **A real `make mile` sim run completes** (you run it) — the loop must be confirmed end-to-end
   with the scripted intervener before swapping in the human device.
3. **A genuinely mediocre base policy** (see §5.4) — MILE is pointless if the base policy already
   succeeds.

## 4. Scope

**In scope:** live MuJoCo rendering; SpaceMouse device passthrough + read; `intervener: spacemouse`
wiring verification; **a gamepad (Xbox/PS) joystick as an alternate teleop device (§5.6)** — the
device the original MILE paper used; mediocre base policy; `COST_LOOKUP` calibration; an N-round
run measured for improvement.
**Out of scope:** AprilTag/real-FR3/Vive (phases b/c); any change to the abstract teleop/intervener
interfaces (they already work — this phase only exercises them).

## 5. Components

### 5.1 Live MuJoCo rendering (`make sim-gui`)

The docker path renders headless to `Xvfb :99` + ffmpeg capture. A human needs to *see* the robot
to decide when to intervene, so add a **live** path that points the container at the host X server:

- Host (once/login): `xhost +local:root`; note `$DISPLAY` (e.g. `:0`/`:1`).
- `docker/docker-compose.yml`: keep `network_mode: host`, mount `/tmp/.X11-unix`, and allow the
  host `DISPLAY` to pass through (a compose override or a `DISPLAY` env that defaults to `:99` but
  can be set to the host display).
- New verb `make sim-gui`: runs `sim_up.sh` **without** starting Xvfb and with `DISPLAY=<host>` so
  the multipanda `mujoco_ros` GLFW viewer opens a real window. With an NVIDIA GPU + container
  toolkit it is hardware-accelerated; otherwise `llvmpipe` software GL (~14 FPS) still shows a
  window.
- **Forward-compat:** keep the headless `make sim-up` path intact for CI/recording — `sim-gui` is
  an alternate entry, not a replacement. The decision of X11-passthrough vs VNC depends on whether
  the lab box has an attached display; X11 is the default, VNC (x11vnc over Xvfb) is the fallback
  for a headless server. **Open decision — see §8.**

### 5.2 SpaceMouse passthrough + read

- Image already carries `pyspacemouse` + `hidapi` (docker increment).
- `docker/docker-compose.yml`: uncomment a `/dev/hidraw*` device mapping pointing at the
  3Dconnexion node (find it via `lsusb` / `ls -l /dev/hidraw*`). If the container user cannot read
  it, either add `privileged: true` or a host udev rule for the device VID/PID.
- **Sanity check before MILE:** a one-liner in `make shell` that opens `pyspacemouse` and prints
  nonzero deflection when the puck is pushed. The whole phase is blocked until this works.

### 5.3 Intervener wiring (already present — verify, don't rebuild)

`scripts/train_mile.py` already maps `intervener: spacemouse` → `TeleopIntervener(SpaceMouseDevice())`,
and `TeleopIntervener.intervene` returns `(action, intervene, done)` straight from
`SpaceMouseDevice.read()`. Work here is **verification + tuning**, not new code:
- Confirm `SpaceMouseDevice` maps deflection past `deadband` (default 0.1) → `intervene=True` (ν=1),
  release → ν=0; that `translation_scale` (default 0.02 m/unit) feels right; that a button maps to
  the gripper and `done`.
- Tune `translation_scale`/`deadband` for feel (parent spec §10). Mirror Vive's clutch semantics so
  phase (c) is a drop-in (**forward-compat**: do not bake SpaceMouse-only assumptions into the
  mapping — keep the `TeleopDevice.read()` contract device-agnostic).

### 5.4 Two collect modes — mediocre (base policy) and perfect (reference)

Resolved: there are **two** collect verbs, not one.
- **`make collect-mediocre`** (`--mediocre true --require_success false`) → `sim_demos_mediocre.npz`.
  This **feeds the base policy** (`make base-policy` consumes it), giving the deliberately mediocre
  policy MILE needs — one that starts the task but cannot reliably finish (parent spec §5.5).
- **`make collect-expert`** (`--mediocre false --require_success true`) → `sim_demos_expert.npz`.
  Perfect, successful-only demos kept as a **reference/upper-bound** (e.g. an expert BC policy to
  compare the MILE-improved policy against, or eval seed). Not the base policy.

After training, **measure** the base policy's sim success rate to confirm it is clearly mediocre
(well below ~60%, visibly failing the precision step). If the mediocre scripted demos still yield
too-competent a policy, increase the degradation (more action noise / a fixed wrong release-height
offset in the mediocre scripted policy). Document the resulting success rate.

### 5.5 `COST_LOOKUP` calibration

`Franka-Stack-Sim-v0` currently uses `[250, 200.0]` (seeded from `pick-place`). After a SpaceMouse
session, compare the **model's predicted intervention probability** against the **human's actual
intervention rate** and adjust `[cost, cdf_scale]` so the computational model's `p(ν=1|s)` tracks
when the human actually took over. Record the tuned values (they are re-tuned again on the real FR3
in phase b, but a sim-calibrated starting point de-risks that).

### 5.6 Gamepad joystick variant (`intervener: joystick`)

The original MILE paper drove interventions with a **gamepad**, not a SpaceMouse. Supporting one
here gives a second real-human device on the *same* loop and a familiar control scheme. Because the
teleop layer is already device-agnostic (`TeleopDevice.read()` → `TeleopReading(action, intervene,
done)`, adapted by `TeleopIntervener`), this is **one new device class + one wiring branch**, with
**no change** to the abstract interfaces, the `Collector`, or the Box dataset.

- **New unit `mile_franka/teleop/joystick.py` — `JoystickDevice(TeleopDevice)`.** Mirrors
  `SpaceMouseDevice`'s structure exactly: an **injectable raw reader** so the mapping logic loads and
  tests with no hardware; the default reader lazily `import pygame` (already in
  `docker/requirements-mile.lock.txt`, `pygame==2.6.1` — **no new dependency**), runs
  `pygame.init()` / `pygame.joystick.init()`, and opens joystick `0`. Constructor args:
  `translation_scale` (default `0.02` m/unit, matching SpaceMouse), `deadband` (default `0.1`), and
  **configurable button/axis indices** (pads differ) with Xbox defaults.
- **Mapping (clutch semantics, Xbox layout defaults):**
  | Input | Effect |
  |---|---|
  | **RB / right bumper (hold)** | clutch engaged |
  | Left stick X / Y | `dx`, `dy` (× `translation_scale`, deadband-gated) |
  | Right stick Y (or a trigger axis) | `dz` (× `translation_scale`, deadband-gated) |
  | **A button (hold)** | gripper close (`+1`); released → open (`-1`) |
  | **Start button** | `done=True` |
- **ν = clutch OR motion (resolved).** `intervene` is True when the clutch is held **OR** any mapped
  axis exceeds the deadband **OR** the gripper button is held — a superset of `SpaceMouseDevice`'s
  `moved or gripper_close` rule, plus an explicit clutch button for deliberate takeovers. When no
  signal is present, `intervene=False` and the action is zeroed. (This is more forgiving than
  clutch-only; if accidental stick drift triggers spurious ν=1 during a session, raise `deadband` or
  switch to clutch-only — a one-line change.) Output is the same `[dx, dy, dz, gripper]` 4-DoF array
  and `TeleopReading`, so the collector and dataset schema are byte-for-byte identical to SpaceMouse.
- **Wiring — `scripts/train_mile.py`:** add a third branch next to the existing two:
  `elif which == 'joystick': from mile_franka.teleop.joystick import JoystickDevice;
  intervener = TeleopIntervener(JoystickDevice())`. `config_franka.json` stays `spacemouse`;
  selecting the gamepad is a one-word config edit.
- **Passthrough + sanity check.** pygame's SDL joystick backend reads `/dev/input/event*` (and/or
  `/dev/input/js0`). Add a **commented** device mapping in `docker/docker-compose.yml` (alongside the
  hidraw SpaceMouse line) so it is config-only, not a rebuild, and a `make joystick-check` one-liner
  mirroring `make spacemouse-check` that prints nonzero deflection / button state when the pad is
  moved. **Forward-compat:** keep the mapping device-agnostic — do not bake gamepad-only assumptions
  into the `TeleopReading` contract; Vive (phase c) remains a drop-in alongside SpaceMouse and this.

## 6. Data flow (unchanged from the existing collector)

`make sim-gui` (live window) → `make mile` with `intervener: spacemouse` **or `joystick`** (same
loop, swappable device). The base policy drives; on puck-deflection past deadband (SpaceMouse) or
clutch-hold / stick-deflection (gamepad) the human action overrides (ν=1), else the policy acts
(ν=0); the
`Collector` records the full Box dataset every step; `InterventionTrainer` retrains; repeat N rounds.
`rollout.auto_eval: false` stays — no autonomous execution.

## 7. Acceptance gates

1. **Live view:** `make sim-gui` shows a live MuJoCo window of the stacking scene.
2. **Device read:** the SpaceMouse sanity check (`make spacemouse-check`) **and** the gamepad
   sanity check (`make joystick-check`) each print nonzero deflection / button state inside the
   container. Either device satisfies gates 4–5.
3. **Mediocre base policy:** measured sim success rate confirms it fails often enough to warrant
   intervention (§5.4).
4. **Human intervention recorded:** a `make mile` round with the human shows ν toggling with puck
   deflection and the human action stored on ν=1 steps (inspect the saved Box dataset).
5. **Improvement:** across N=2–3 rounds the policy's success rate rises (human-judged or a guarded
   eval), with `COST_LOOKUP` calibrated so predicted vs actual intervention rates roughly match.

## 8. Open decisions (for the user)

1. ~~Mediocre base policy strategy~~ — **resolved (§5.4):** two collect modes; base policy trains on
   `collect-mediocre`, `collect-expert` kept as a reference. Still measure the result to confirm it
   is mediocre enough.
2. **Live-render transport** — X11 passthrough (default; needs an attached display on the lab box)
   or VNC-over-Xvfb (for a headless server)? Depends on your lab box.
3. **GPU** — is there an NVIDIA GPU on the lab box (hardware GL + faster training), or do we stay on
   software GL + CPU torch?
