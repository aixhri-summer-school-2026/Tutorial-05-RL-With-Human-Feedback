# Part 3 — MILE on the Franka Sim (you are the human)

**Goal:** this is the core experience. You boot the MuJoCo Franka simulator, watch the base
policy try to stack cubes, and **step in with the keyboard when it's about to fail**. Your
interventions — collected through the same `mile_cont_loss_fn` you wrote — train your own
policy. In Part 1 a synthetic expert played the human; here *you* are the expert.

**Prereqs:** [03-metaworld.md](03-metaworld.md) done, [04-franka-fake.md](04-franka-fake.md)
prints `smoke_franka_env ok`.

> **A note on runtime:** unlike MetaWorld, this part is paced by **ROS + MuJoCo + you**, not
> by training. Booting the controller stack takes ~90 s (all DDS discovery and controller
> gains), each episode is real-time teleop, and training on your handful of episodes is
> seconds. There's no meaningful way to "speed it up" — the wall-clock is dominated by ROS
> bring-up and human input, both fixed costs.

---

## The environment: `Franka-Stack-Sim-v0`

Same task and interface as the fake backend ([04-franka-fake.md](04-franka-fake.md)), but now
running against hucebot's **`multipanda_ros2` MuJoCo simulation** driven by a real
**Cartesian-impedance controller** — the *same* controller code that runs on the hardware in
[06-franka-real.md](06-franka-real.md).

| | |
|---|---|
| Backend | `MultipandaRosBackend` → MuJoCo via ROS 2 |
| Pose source | `MujocoGtPoseSource` (ground-truth cube poses from the sim) |
| **Action** | 4-D: `[Δx, Δy, Δz, gripper]` |
| **Observation** | 9-D reduced → `FrameStack(10)` → **90-D** |
| Cubes | 5 cm; `table_z = 0.0` (matches the multipanda stacking scene) |
| Success | cubes stacked + stable → `info['success'] == 1` |
| Intervention cost | `COST_LOOKUP['Franka-Stack-Sim-v0'] = [2, 2.0]` |

The cost `[2, 2.0]` is much smaller than MetaWorld's `[75, 175.0]` because the Franka base
policy is a **BC `ActorCriticPolicy`** whose log-probs sit around `[+4, +7]` (no tanh
squashing), versus MetaWorld SAC's `[-30, 0]`. Same model, different log-prob scale → different
cost. (See `mile/computational_model.py` for the note.)

---

## The base policy (and how it was made)

You start from `trained_models/franka/base_policy` — a **deliberately mediocre** policy so
there's something for you to correct. It was produced offline, before the tutorial, by:

1. `make collect-mediocre` — a **scripted** stacking policy (`ScriptedStackPolicy`, a state
   machine over ground-truth cube poses with randomized transit/lift heights) rolls out and
   records demos to `sim_demos_mediocre.npz`. "Mediocre" = it includes sloppy, sometimes-failing
   attempts, not just perfect ones.
2. `make base-policy` — **behavior cloning** (`imitation`'s BC) fits an `ActorCriticPolicy`
   (net `[256, 256]`, `NormalizeFeaturesExtractor` + `RunningNorm`) to those demos.

The result stacks successfully **some** of the time and fails the rest — exactly the regime
where human intervention teaches the most.

---

## Step 1 — Boot the simulator

From your **host** (not inside the container):

```bash
make sim-up          # launch the MuJoCo stacking sim headless (wait ~20 s)
```

Optional live viewer (host display; run `xhost +local:root` once per login first):

```bash
xhost +local:root
make sim-gui
```

A MuJoCo window shows the Franka arm and two red cubes. If rendering is slow or the window
won't open, see [08-troubleshooting.md](08-troubleshooting.md) — you can run everything
headless without the viewer.

---

## Step 2 — Learn the keyboard (free play)

Inside `make shell` (a container terminal):

```bash
make tutorial-teleop      # free-play; no data saved, Ctrl-C to exit
```

Two windows open: the large **MuJoCo viewer** (watch the arm — don't type here) and a small
black **"MILE keyboard teleop"** window (**click it to give it focus before pressing keys**).
The MuJoCo viewer steals keys like `space` (pause) and `w`/`a` (visualization) if it's focused,
so always click the teleop window first.

| Key | Action |
|---|---|
| `w`/`s` | forward/back (Δx) |
| `a`/`d` | left/right (Δy) |
| `q`/`e` | up/down (Δz) |
| **SPACE** | **toggle intervention (ν)** — clutch on/off |
| `g` | toggle gripper |
| `enter` | end episode (save) |
| `backspace` | discard episode |

Full reference and the joystick map: **[07-teleop.md](07-teleop.md)**.

**The clutch (SPACE) is the whole idea:** when ν=0 the *policy* drives and nothing is
recorded; press SPACE to take over (ν=1) and every step you command is recorded as a human
correction; press SPACE again to hand control back. Practice picking up a cube and stacking
it a couple of times until the controls feel natural.

> **First-run patience:** the teleop/collect commands take ~90 s to connect the first time
> (ROS bring-up + controller gains). This is normal; it is not frozen. See
> [08-troubleshooting.md](08-troubleshooting.md).

---

## Step 3 — Measure the base policy (before)

Stop the free-play teleop, then in the container:

```bash
make eval-base       # 5 autonomous episodes, no human
```

**Example output:**
```
episode 0: success=1 steps=138
episode 1: success=0 steps=1000
episode 2: success=1 steps=97
episode 3: success=0 steps=1000
episode 4: success=1 steps=688

BASE POLICY SIM SUCCESS RATE = 0.60 (3/5); mean steps=585
```

`success=0` means the episode timed out (failed). **Write down this before-score.** It
varies a lot run to run — the base policy is genuinely mediocre, which is the point. A
measured 5-episode run on this repo scored **0.40 (2/5)**; the example above shows a luckier
0.60. Anywhere in the ~0.4–0.6 range is normal.

---

## Step 4 — Collect your interventions and train

```bash
make tutorial-collect-train      # inside make shell (needs the keyboard window)
```

This runs `tutorial_train.py --config config/tutorial_franka.yaml`: **1 round, 2 episodes
collected, 100 training epochs**, using *your* loss.

**The loop, per episode:**
1. The policy drives (ν=0). Watch it.
2. When it's about to fail — reaching past a cube, gripper off-target, drifting — press
   **SPACE** and take over.
3. Guide it to fix the situation (nudge into place, toggle the gripper with `g`).
4. Press **SPACE** again to release control back to the policy.
5. **ENTER** ends the episode and saves; **BACKSPACE** discards it to retry.

**How much:** 2 episodes with 1–2 interventions each is enough. You're teaching the policy to
handle the specific failure modes you saw — you don't need to fly it perfectly.

**What trains:** the same two-term loss from Part 1 — BCE on the intervention flag ν (all
steps) + Gaussian NLL of *your* action on the ν=1 steps (the corrections). The policy learns
your fixes; the mental model learns when you chose to step in.

---

## Step 5 — Measure the trained policy (after)

```bash
make eval-mile       # 5 autonomous episodes with the MILE-trained policy
```

**Example output:**
```
episode 0: success=1 steps=112
episode 1: success=1 steps=245
episode 2: success=0 steps=1000
episode 3: success=1 steps=98
episode 4: success=1 steps=310

BASE POLICY SIM SUCCESS RATE = 0.80 (4/5); mean steps=353
```

> The label always reads "BASE POLICY SIM SUCCESS RATE" — it's the script's fixed header;
> here it's evaluating your MILE policy.

**The delta:** 0.60 → 0.80 = **+0.20**. Interpretation:

- **Δ > 0.10:** your interventions clearly taught the policy. 🎉
- **Δ ≤ 0.05:** normal from sparse data — the base policy was already close, or your
  interventions didn't hit the failure mode that eval happened to sample. This is *best-effort*;
  MetaWorld (Part 1) is the *guaranteed* improvement demo because it uses dense synthetic data.

---

## Generate more videos of the final policy

`eval-base`/`eval-mile` fix the episode count at 5. To render **more** eval videos of a
specific saved run (e.g. for a slide or to inspect the corrections you taught), use:

```bash
make eval-run                                   # 10 episodes of output_dir/franka/policy
make eval-run POLICY=output_dir/franka/policy EPISODES=20
```

It writes one `.mp4` per episode under `output_dir/franka/eval_run_videos_<timestamp>/`.
`eval-base`/`eval-mile` also drop videos, but `eval-run` lets you point at any run directory
and pick how many.

---

## What you can tune (and what to expect)

`config/tutorial_franka.yaml` unless noted.

| Knob | Higher → | Expected effect |
|---|---|---|
| `episodes_per_round` | more of your episodes per round | more human signal, more of your time |
| `num_rounds` | more collect→train cycles | more improvement, but each round is more teleop |
| `num_epochs` | more gradient steps | better fit to your data; too high overfits ~hundreds of steps |
| `keyboard_translation_scale` | bigger per-key motion | coarser/faster teleop; smaller = finer control |
| `intervener` | `keyboard` / `joystick` | input device (see [07-teleop.md](07-teleop.md)) |
| `cost` in `COST_LOOKUP` | human "should" intervene less | shifts the modeled intervention threshold; retune if you swap the base-policy architecture |

### Extension: ablate the loss weights

Your loss is `λ₁·NLL(action) + λ₂·BCE(intervention)`, both `1.0` by default. Edit `lambda1`/
`lambda2` under `train:` in the config:

| λ₁ | λ₂ | Expected effect |
|----|----|-|
| 1.0 | 1.0 | baseline |
| 0.0 | 1.0 | policy never learns your *actions* — only *when* you'd intervene |
| 1.0 | 0.0 | policy learns your actions but ignores ν — plain BC on your corrections |
| 2.0 | 0.5 | prioritize action-matching; softer intervention classifier |

Which do you expect to hurt most? Run it and see. (This applies identically in Part 1 —
`config/tutorial_metaworld.yaml`.)

---

## The takeaway

> One loss, two worlds. The `mile_cont_loss_fn` you wrote trained a MetaWorld policy from a
> synthetic expert (Part 1) and *your* policy from *your* keyboard corrections (here) — same
> code, no changes. Next you'll run it on real hardware.

## Next step

→ **[06-franka-real.md](06-franka-real.md)** — the same loop on a live FR3 (sim→real payoff).
