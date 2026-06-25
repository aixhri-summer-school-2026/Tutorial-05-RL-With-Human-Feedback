# Teleop Reference: Controlling the Robot

This guide explains how to control the Franka robot during the tutorial using keyboard or joystick. The key concept is **segment-toggle intervention**: you decide when you're "in control" by toggling a clutch button, and the robot learns from both these moments and the times you let it run autonomously.

---

## The clutch metaphor: ν (nu) segments

When you intervene, you're setting **ν = 1** — the "intervention flag" that MILE uses to learn. Think of it like a car clutch:

- **Press the clutch (toggle ON)**: you take over. The robot records your actions and marks them as interventions (ν=1).
- **Release the clutch (toggle OFF)**: the robot runs under its policy. No data is recorded.
- **Toggle again**: you're back in control.

Each press/release of the intervention button toggles your state. Once in a segment (ν=1), every timestep of your control is recorded. Exit the segment, and the robot runs free until you toggle back in.

**Gripper state**: Initially it will pass through whatever the policy outputs (NaN). The first time you toggle the gripper button, it locks in to either open (-1) or closed (+1). After that, each gripper toggle switches between the two states.

---

## Keyboard (default for Parts 2 & 3)

The keyboard is the default teleop device for the tutorial's sim and fake-world parts. Use it for development, testing, and Part 3 (Franka in MuJoCo).

**Keep the keyboard window focused** while controlling — only the focused window receives input, so click the small "MILE keyboard teleop" window before pressing keys. (An opt-in OS-wide listener exists; see the tips below.)

| Key | Action | Notes |
|---|---|---|
| **W** | Move forward (+X) | 0.02 m per press |
| **S** | Move backward (-X) | 0.02 m per press |
| **A** | Move left (+Y) | 0.02 m per press |
| **D** | Move right (-Y) | 0.02 m per press |
| **Q** | Move up (+Z) | 0.02 m per press |
| **E** | Move down (-Z) | 0.02 m per press |
| **Space** | Toggle intervention segment | Press once to start (ν=1), again to stop (ν=0) |
| **G** | Toggle gripper | First press: lock open or closed; subsequent presses: toggle between states |
| **Enter** | End episode (success) | Marks the current rollout as complete; soft reset |
| **Backspace** | Discard episode | Rejects all data from this attempt; starts fresh |

### Keyboard tips

- **Focus the teleop window first.** The MuJoCo viewer (`make sim-gui`) has its own key bindings that overlap with the teleop keys: `w`=wireframe, `a`=axis, `s`=skeleton, `q`=visualization toggle, `e`=contacts, **`space`=pause simulation**. If the MuJoCo window is focused, pressing these keys changes the visualization or pauses the sim instead of moving the robot. Always click the small black "MILE keyboard teleop" window before pressing any keys.
- **Optional OS-wide capture**: construct `KeyboardDevice(reader_backend="global")` to read keys regardless of focus (no window to hunt for). Caveat: a focused MuJoCo viewer *also* receives the keys, so its bindings fire on top of teleop (space pauses the sim). Only useful if you keep the terminal focused, never the viewer. The default `"pygame"` backend avoids this entirely.
- **Debounce protection**: rapid key repeats are filtered (200 ms lockout). A quick tap registers once.
- **Know when you're intervening**: watch the terminal log. When ν=1, you'll see messages like `intervention_prob: 0.95` in the collector output.
- **Test motion first**: press a movement key once. If nothing happens, the window isn't focused. Click it and try again.
- **Exit before save**: press Backspace to discard, or Enter to save the rollout (only ν=1 segments are kept in the dataset).

---

## Joystick / Xbox Gamepad (`intervener: joystick`)

Xbox 360, Xbox One, and compatible gamepads work out of the box. Select this device in `config/franka_sim.yaml`:

```yaml
experiment:
  intervener: joystick
  # ... other settings
```

### Button & stick map

| Control | Action | Notes |
|---|---|---|
| **Left stick X/Y** | Move X/Y | Proportional to stick deflection; deadband 0.1 |
| **Right stick Z** | Move Z | Right stick up = +Z (lift); down = -Z (lower) |
| **RB (right bumper)** | Toggle intervention segment | Press to enter/exit ν=1 mode |
| **A (bottom face)** | Toggle gripper | Alternates between open (-1) and closed (+1) |
| **Start** | End episode (success) | Saves the rollout |
| **Back (select)** | Discard episode | Rejects current data; starts fresh |

### Joystick tips

- **No focus requirement**: works without a window in focus (unlike keyboard).
- **Stick deadband**: small deflections (<0.1) are ignored to prevent drift.
- **Clutch philosophy**: RB is a toggle, not a hold. Press once to start intervening (ν=1); press again to stop (ν=0).
- **Translation scale**: default is 0.5 m per unit of stick deflection (configurable in config via `"joystick_translation_scale"`).

---


## How to know you're intervening (ν=1)

As you control the robot, watch the terminal/collector output. You'll see lines like:

```
[Collector] step 42, intervene=True, intervention_prob=0.87, ...
[Collector] step 43, intervene=True, intervention_prob=0.92, ...
[Collector] step 44, intervene=False, intervention_prob=0.15, ...  # <- you released the clutch
```

- **intervene=True** → ν=1, your actions are being recorded.
- **intervene=False** → ν=0, the policy is running; no intervention data recorded.

---

## Workflow: a typical rollout

1. **Start the controller** (keyboard or joystick).
2. **Watch the robot plan** (ν=0, you're not intervening). It might reach toward the wrong cube or move awkwardly.
3. **Press the clutch toggle** (space or RB) to enter a segment (ν=1).
4. **Guide the robot** with stick/keyboard/hand input. Use movement keys to nudge, and gripper toggle when you need to open/close.
5. **Release the clutch** (press toggle again) once you've corrected the error. The robot resumes autonomy (ν=0).
6. **Repeat** for the next error, or press **Enter** / **Start** to end the episode successfully.

At the end of an episode:
- Press **Enter** (keyboard) or **Start** (joystick) to **save** the rollout with your intervention(s).
- Press **Backspace** (keyboard) or **Back** (joystick) to **discard** and try again.

---

## Configuration: choosing your device

In `config/franka_sim.yaml` (for simulation) or `config/franka_real.yaml` (for hardware):

```yaml
experiment:
  intervener: keyboard    # or "joystick"
  # ... other settings
```

- **Parts 2 & 3 (fake + MuJoCo sim)**: use `keyboard` or `joystick` (no ROS/hardware).
- **Part 4 (real FR3)**: use `joystick` (default).

Switch devices by editing the config and restarting the collector.

---

## Troubleshooting

### Keyboard: keys not registering
- **Check focus**: click the small pygame window (labeled "MILE keyboard teleop — keep focused"). Only the focused window receives keys.
- **Check key layout**: if you're not on QWERTY, remap the keys in code or use joystick instead.

### Joystick: no input detected
- **Check connection**: `ls /dev/input/js*` (or ask the instructor).
- **Test**: press a button. If the terminal shows nothing, the gamepad may not be detected.
- **Fallback**: use keyboard instead.

### "intervene=False" even though I'm holding the clutch
- **Wrong mode**: confirm your config uses `"segment_mode": true` (default). Older configs may use per-frame clutch.
- **Debounce cooldown**: if you toggled too fast, wait 200 ms between presses.

---

## Next steps

→ **[05-troubleshooting.md](05-troubleshooting.md)**: Stuck? Find solutions here.
