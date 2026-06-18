#!/usr/bin/env python3
"""Identify axis/button indices and verify ghost-filter + debounce + segment behaviour.

Tracks ALL buttons independently (not just configured ones) so you can find which
index to assign to clutch, gripper, and done before configuring the device.

For each button event, shows three stages:
  raw DOWN  - physical edge (unfiltered)
  TRIGGER   - accepted after hold_confirm_s hold and debounce_s lockout
  GHOST     - press released before hold_confirm_s (correctly dropped)
  DEBOUNCED - hold was long enough but lockout was still active

Axes show only when they change from their at-rest baseline (suppresses L2/R2 at -1.0).

Run inside the container:  make shell  →  python3 scripts/joystick_identify.py
"""
import time

import pygame

HOLD_CONFIRM_S = 0.05   # must hold this long to confirm (ghost filter)
DEBOUNCE_S = 0.3        # lockout after accepted trigger
AXIS_CHANGE_THRESHOLD = 0.05

pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
js.init()
n_axes = js.get_numaxes()
n_btns = js.get_numbuttons()

# Capture at-rest baseline (suppresses e.g. L2/R2 triggers resting at -1.0)
print("Measuring axis baseline (don't touch the controller)...", end=" ", flush=True)
time.sleep(0.5)
pygame.event.pump()
baseline = [js.get_axis(i) for i in range(n_axes)]
print("done.")
print(f"  baseline: { {i: round(baseline[i], 2) for i in range(n_axes)} }\n")
print(f"{n_axes} axes, {n_btns} buttons")
print(f"ghost filter >{HOLD_CONFIRM_S*1000:.0f}ms hold  |  debounce {DEBOUNCE_S*1000:.0f}ms lockout")
print("Xbox 360 expected: RB(5)=clutch  A(0)=gripper  Start(7)=done")
print("Press buttons to identify indices. Ctrl-C to stop.\n")

# Per-button state (tracks ALL indices, not just configured ones)
prev_raw:       list[bool]        = [False] * n_btns
press_start:    dict[int, float]  = {}   # time button first went down (identify clock)
triggered_this: dict[int, bool]   = {}   # fired once per physical press
last_trigger:   dict[int, float]  = {}   # time of last accepted trigger
prev_axes = list(baseline)

while True:
    now = time.monotonic()
    pygame.event.pump()
    axes = [js.get_axis(i) for i in range(n_axes)]
    raw = [bool(js.get_button(i)) for i in range(n_btns)]

    # --- axes: show only those that moved from baseline ---
    changed = {
        i: round(axes[i], 2)
        for i in range(n_axes)
        if abs(axes[i] - baseline[i]) > AXIS_CHANGE_THRESHOLD
           and abs(axes[i] - prev_axes[i]) > 0.01
    }
    if changed:
        print(f"  axes:    {changed}")
    prev_axes = axes

    # --- buttons: independent ghost+debounce tracking for every index ---
    for i in range(n_btns):
        cur, prev = raw[i], prev_raw[i]

        if not prev and cur:
            # Rising edge
            press_start[i] = now
            triggered_this[i] = False
            print(f"  btn {i:2d}:  raw DOWN")

        if prev and not cur:
            # Falling edge — classify the press
            p_start  = press_start.get(i, now)
            held_ms  = (now - p_start) * 1000
            last_t   = last_trigger.get(i, -999.0)
            fired    = p_start <= last_t <= now   # trigger landed during this press

            if fired:
                fire_ms = (last_t - p_start) * 1000
                print(f"  btn {i:2d}:  UP  (held {held_ms:.0f}ms → TRIGGER at +{fire_ms:.0f}ms)")
            elif held_ms < HOLD_CONFIRM_S * 1000:
                print(f"  btn {i:2d}:  UP  (held {held_ms:.0f}ms → GHOST, too short)")
            else:
                print(f"  btn {i:2d}:  UP  (held {held_ms:.0f}ms → DEBOUNCED, lockout active)")

            press_start.pop(i, None)

        # Advance ghost+debounce state for held buttons
        if cur and not triggered_this.get(i, True):
            held_s     = now - press_start.get(i, now)
            since_last = now - last_trigger.get(i, -999.0)
            if held_s >= HOLD_CONFIRM_S and since_last >= DEBOUNCE_S:
                last_trigger[i]    = now
                triggered_this[i]  = True   # one trigger per physical press

    prev_raw = raw
    time.sleep(0.02)  # 50 Hz for fine-grained hold measurement
