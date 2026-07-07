# MILE Tutorial — Hands-On Flow

This is your step-by-step guide. By now you've set up Docker (`00-setup.md`), read the concepts (`01-concepts.md`), and implemented the three exercises (`02a` → `02b` → `02c`). Now it's time to run the full MILE loop: from a synthetic expert (MetaWorld) through your own interventions (Franka sim) to the real robot.

**All commands run inside the container.** To start:
```bash
make up && make shell
```

---

## Land & Verify

**Goal:** Confirm your setup is ready; get a mental model of MILE.

### Step 1: Run the readiness check

Inside the container, run:

```bash
make tutorial-check
```

**Expected output:**
```
  import metaworld: OK
  import mile: OK
  import mile_franka: OK
  import torch: OK
  import gymnasium: OK
  artifact trained_models/initial_policy: OK
  artifact trained_models/expert_policy: OK
  artifact trained_models/gt_mental_model: OK
  artifact trained_models/warm_started_mental_model: OK
  artifact trained_models/franka/base_policy: OK
tutorial-check OK — you're ready.
```

**If you see any `FAIL` or `MISSING`:** stop and call an instructor. Do not proceed.

### Step 2: Read the concepts and complete the exercises

If you haven't already, work through:

1. **[01-concepts.md](01-concepts.md)** — 5-min MILE primer
2. **[02a-scripted-intervener.md](02a-scripted-intervener.md)** — Exercise 1: when does the human step in?
3. **[02b-loss-exercise.md](02b-loss-exercise.md)** — Exercise 2: implement the training objective
4. **[02c-intervention-model.md](02c-intervention-model.md)** — Exercise 3: design p(ν=1 | state)

All three auto-apply their answer keys when you train — you can move on even if incomplete.

---

## MetaWorld: Your Loss on the Paper Benchmark

**Goal:** Watch MILE train on the classic peg-insertion task with the loss you just wrote.

No human involved — you're watching an *automated synthetic expert* intervene. The point is to validate that your loss works on clean sim signal.

```bash
make tutorial-metaworld
```

**Duration:** ~5–8 min on GPU, ~12–20 min on CPU. If you see loss values scrolling, it's running — not stuck.

> **Start sim-up in parallel:** while this trains, open a second terminal on your host and run `make sim-up`. That way the Franka sim is already booting when MetaWorld finishes.

**What to expect:**

The script first prints three `[tutorial]` lines confirming which implementations are active (yours or the answer key). Then per-batch and per-epoch loss tables scroll, followed by a summary at each evaluation checkpoint. Exact numbers vary by run (±0.2 is normal), but you should see the `success_rate` row **increase from Round 0 to Round 1**:

```
Round: 0
...
| mile/                |       |
|    success_rate      | 0.40  |
Round: 1
...
| mile/                |       |
|    success_rate      | 0.60  |
```

If the loss is not yet implemented, you'll see: `[tutorial] your loss is missing/incorrect — applying the answer key so you can train.` — this is expected and training continues with the reference solution.

If success does **not** increase at all, your loss may have a bug — re-run `make tutorial-check-loss`.

**What's happening:**

1. Round 0: the synthetic human (downloaded expert + MILE's intervention model) collects data as the initial policy tries and fails.
2. The policy trains on interventions — your loss guides it.
3. Round 1: the improved policy collects data. Success climbs further.

**The key insight:** "Success improved from Round 0 to Round 1. I wrote the loss. My loss made this happen."

This is your proof that the loss is correct *and* that MILE works. In the next step, you'll replace the synthetic expert with your own keyboard interventions.

### Extension: Tune the Intervention Cost

The intervention model uses two per-task parameters in `COST_LOOKUP` (`mile/computational_model.py`):

```python
COST_LOOKUP = {
    'peg-insert-side-v2': [75, 200.0],
    'Franka-Stack-Sim-v0': [2, 2.0],
    ...
}
```

`cost` shifts the CDF threshold: higher cost → human intervenes less often (policy needs to diverge more before p(ν=1|s) rises). `cdf_scale` controls the sharpness of the CDF transition.

**Try it:** open `mile/computational_model.py` and change `'peg-insert-side-v2': [75, 200.0]` to `[150, 200.0]` (double the cost). Re-run `make tutorial-metaworld`. Does the success rate still improve? Does it improve faster or slower?

**Questions to consider:**
- How do you choose cost for a *new* task you've never run before?
- If p(ν=1|s) is near 0 everywhere, what happens to the discrete loss term?
- Is there a cost value that makes MILE equivalent to standard BC (just imitating interventions)?

---

## Franka Fake Backend: Environment Smoke Test

**Goal:** Confirm the Franka environment loads and runs end-to-end on the simplest backend (no physics, no ROS).

```bash
make tutorial-fake
```

**Expected output:**
```
smoke_franka_env ok
```

This runs one scripted episode on the **fake backend** — a simple kinematic world with no MuJoCo physics — and asserts success. If it prints `ok`, you're clear to run the full sim. If it fails, call an instructor before proceeding.

The observation/action dimensions you'll use in the sim:
- **Obs:** 9-dim (EE position, gripper width, cube position) × `FrameStack(10)` = 90-dim.
- **Action:** 4-DoF (Δx, Δy, Δz, gripper command).

---

## Visual Sim & Learn the Keyboard

**Goal:** Start the MuJoCo sim and practice the teleop controls.

### Step 1: Launch the headless simulator

From your **host machine** (not inside the container), start the simulation in the background:

```bash
make sim-up
```

Wait ~20 seconds for the sim to boot. (The first time is slower; subsequent restarts are faster.)

### Step 2: Open the viewer on the host

On your **host machine** (laptop, not inside the container), allow X11 access (once per login) and open the viewer:

```bash
xhost +local:root   # one-time, allows the container to draw on your display
make sim-gui
```

A MuJoCo window opens. You'll see:
- The Franka robot (light blue arm) at home.
- Two red cubes on the table (the stack target).
- A 3D viewport.

**Note on rendering:** If the window is slow or doesn't appear, see [05-troubleshooting.md](05-troubleshooting.md#gl-rendering-slow). You can skip the visual and proceed headless using `make eval-base` and `make tutorial-collect-train` without `sim-gui`.

### Step 3: Learn the keyboard (free-play teleop)

**Inside `make shell`** (the container — open a new terminal tab), run:

```bash
make tutorial-teleop
```

This opens **two windows**:

- **MuJoCo viewer** (large 3D window, from Step 2): shows the robot. Watch this to see the arm move. **Do not type here.**
- **Teleop window** (small black pygame window labeled "MILE keyboard teleop"): the keyboard input receiver. **Click this window before pressing any keys.**

> ⚠️ **Critical:** The MuJoCo viewer has its own key bindings that overlap with the teleop keys — `w`=wireframe, `a`=axis display, `q`=visualization toggle, **`space`=pause simulation**, etc. Only the focused window receives keystrokes, so if the MuJoCo window is focused when you press these keys, the simulation will toggle visualization modes or pause instead of moving the robot. Always click the small **teleop window** first.

Refer to **[04-teleop.md](04-teleop.md)** for the full key map, but here's the cheat sheet:

| Key | Action |
|---|---|
| `w` / `s` | move forward/backward (Δx) |
| `a` / `d` | move left/right (Δy) |
| `q` / `e` | move up/down (Δz) |
| **SPACE** | **toggle intervention (ν)** — press once to take over, press again to release |
| `g` | toggle gripper (open/close) |
| `enter` | end episode (done) |
| `backspace` | discard episode (undo) |

**What to practice:**

1. Move the arm to a cube with **w/a/s/d/q/e**.
2. Close the gripper (`g`) to pick it up.
3. Move to the stack location.
4. Open the gripper to place it.
5. Repeat — try to stack both cubes.

**The clutch toggle (SPACE):**

- When **ν=0** (clutch off), the policy is commanding the arm. You see the policy's motion in real-time.
- When **ν=1** (clutch on), **you** command the arm. The policy learns from this.
- In the next step, you'll toggle the clutch on *only when the policy is about to fail*, and toggle it off once you've corrected the error.

---

## Be the Human in Sim

**Goal:** Collect your own interventions, train on your data, watch your policy improve.

**This is the core experience:** you replace the synthetic expert. Your interventions teach your own policy.

### Step 1: Evaluate the base policy (before)

Stop the free-play teleop. In the container, run:

```bash
make eval-base
```

**Expected output:**
```
episode 0: success=1 steps=138
episode 1: success=0 steps=1000
episode 2: success=1 steps=97
episode 3: success=0 steps=1000
episode 4: success=1 steps=688

BASE POLICY SIM SUCCESS RATE = 0.60 (3/5); mean steps=585
```

(Episodes are 0-indexed. `success=1` = succeeded; `success=0` = timed out. Numbers vary between runs.)

**Record the before-score.** In this example, it's **0.60** (60% success).

### Step 2: Collect interventions with keyboard teleop

Inside **`make shell`**, run the combined collect-and-train command:

```bash
make tutorial-collect-train
```

**What happens:**

1. The training system launches.
2. The sim displays the arm. The policy is running.
3. **You watch and intervene:** when you see the policy is about to fail (reaching for empty space, gripper off-target, etc.), **press SPACE** to toggle intervention on and take over.
4. Once you fix the situation, **press SPACE again** to toggle intervention off and let the policy continue.
5. When the episode ends (cube stacked or failed), press **ENTER** to mark the episode done. Press **BACKSPACE** to discard if you want to retry.

**How much to collect:**

- Aim for **2–3 episodes** with 1–2 interventions per episode.
- Don't over-engineer — "good enough" is fine. You're teaching the policy to handle realistic human feedback.

**What you're training:**

The same loss you implemented for MetaWorld, now on *your* data. The policy learns from your interventions (ν=1) and respects your non-interventions (ν=0).

### Step 3: Evaluate the trained policy (after)

After training finishes, run:

```bash
make eval-mile
```

**Expected output:**
```
episode 0: success=1 steps=112
episode 1: success=1 steps=245
episode 2: success=0 steps=1000
episode 3: success=1 steps=98
episode 4: success=1 steps=310

BASE POLICY SIM SUCCESS RATE = 0.80 (4/5); mean steps=353
```

> **Note:** The label "BASE POLICY SIM SUCCESS RATE" appears regardless of which policy is evaluated — it's the script's fixed output format. Here it's showing the MILE-trained policy's rate.

**Record the after-score.** In this example, it's **0.80**.

**The delta:** 0.60 → 0.80 = **+0.20 (20% improvement)**.

**Interpret the result:**

- **Visible improvement** (Δ > 0.10): great! Your interventions directly taught the policy.
- **Small improvement or no change** (Δ ≤ 0.05): the base policy was already close to optimal, or your interventions didn't cover the failure mode. This is normal from sparse data; the *guaranteed* improvement demo is MetaWorld, which removes the pressure on Franka's sparse interventions.

**Review the videos:**

Episode videos are saved to `output_dir/franka/eval_mile_videos_*/`. Watch them — you'll see the policy executing the corrections you taught it.

---

## Real FR3 (Parallel Queue)

**Goal:** Rotate through the live robot and experience sim→real transfer.

**This happens in parallel with the sim step.** You don't wait for the entire room to finish. Instead, staff manage a queue at the real-robot station.

### When it's your turn:

**Location:** the real-robot station (staffed by instructors).

**Setup (staff will handle):**

1. Confirm the FR3 is powered and homed.
2. Mount cubes on the table.
3. Confirm camera calibration (`MILE_CAMERA_CALIB`).

**Your experience:**

- **Same keyboard** as the sim step.
- **Same policy** trained in your laptop's sim.
- **Same MILE loop** — collect interventions, retrain, eval.

**Commands (staff runs these):**

```bash
# Launch the FR3 controller (staff)
make franka-up

# In parallel, staff launch camera + AprilTag (staff)
make apriltag-up

# You collect interventions with keyboard
make tutorial-collect-train
```

**The cubes:**

Real wooden cubes are tracked via AprilTag markers (vision) instead of MuJoCo ground truth. Everything else is identical to the sim step.

**What's identical (sim→real):**

- The Cartesian-impedance controller (same code, same stiffness).
- The MILE loss (your implementation).
- The policy net architecture.
- The keyboard teleop interface.

**What changed:**

- **Backend:** multipanda ROS2 sim → real FR3 (hucebot controller).
- **Pose source:** MuJoCo ground truth → AprilTag camera measurements.

This is the **sim→real transfer payoff**: the sim taught you how MILE works; the real robot proves it works on hardware.

---

## Wrap & Q&A

**Key takeaways:**

1. **One loss, multiple tasks:** you wrote `mile_cont_loss_fn` once, and it trained the MetaWorld policy, the Franka sim policy, and the real-robot policy. That's the power of MILE — the method is task-agnostic.

2. **Sim→real is one backend swap:** the same robot controller ran in sim and on hardware. The only differences were the observations (sim ground-truth vs. real AprilTag).

3. **You are the expert:** in MetaWorld, a synthetic expert improved the policy. In the sim and real-robot steps, *you* were the expert. MILE learns from human intuition — when you see the policy about to fail, you step in.

4. **Joint training matters:** the policy and mental model train together. Your interventions teach both: the policy learns the right action, and the mental model learns when you'd intervene.

**Extension: ablate the loss weights**

Your combined loss is `λ₁ × NLL(action) + λ₂ × NLL(intervention)`. Both default to 1.0. Try these ablations in `config/tutorial_metaworld.yaml` (look for `lambda1`/`lambda2` under `train:`):

| λ₁ | λ₂ | Expected effect |
|----|----|-|
| 1.0 | 1.0 | Baseline (default) |
| 0.0 | 1.0 | Policy never learns from action corrections — only learns *when* humans step in |
| 1.0 | 0.0 | Policy learns the right action but ignores the intervention flag — just BC on ν=1 steps |
| 2.0 | 0.5 | Prioritize action prediction; softer intervention classifier |

Which ablation do you expect to hurt most? Run it and find out.

**Talking points (instructor will cover):**

- Why the intervention flag ν is central (not just mimicry, but *when* to ask for help).
- How the probit intervention model decides when p(ν=1|s) is high enough to ask.
- Why MetaWorld's guaranteed improvement is separate from the Franka sim's best-effort (sparse data).
- The real-robot transfer: what worked, what surprised you, what failed.

**Questions?**

Ask the instructors. They've run this loop many times. Common questions:

- "Why did my policy improve less than MetaWorld?" → MetaWorld uses a synthetic expert with dense data; your Franka interventions are sparse. Both are valid; MetaWorld is the guarantee.
- "Can I try more rounds?" → Yes, run `make tutorial-collect-train` again.
- "What if I want to deploy this on a different robot?" → The code is robot-agnostic; you'd swap the backend (RobotBackend ABC) and pose source.

---

## Troubleshooting During the Tutorial

**Quick fixes for common issues:**

### "sim-up" hangs or doesn't respond

```bash
# (in container) Kill any leftover sim process
pkill -f "sim_up\|mujoco_ros\|franka_sim"
# Try again
make sim-up
```

### sim-gui doesn't open (GL errors on SSH)

```bash
# (on host) Allow local X11 access
xhost +local:root

# Then retry
make sim-gui
```

If still stuck, fall back to headless (skip sim-gui, proceed directly to eval-base).

### Keyboard doesn't respond in sim-gui

Click the small "MILE keyboard teleop" window to focus it (only the focused window receives keys), then try again. If the sim is running (verify with `ros2 topic list | grep sim`), the teleop should work.

### "make tutorial-collect-train" hangs at "waiting for policy…"

The policy import may be loading. Wait 30 sec. If it doesn't respond, Ctrl-C and check:

```bash
# (in container) Verify the policy exists
ls -la trained_models/franka/base_policy
```

If missing, run `make tutorial-fake` first to populate it.

### eval-base shows 0% success

This is rare. Confirm:

```bash
# (in container) Verify the env can load
python3 -c "from mile_franka import make_franka_env; env = make_franka_env('Franka-Stack-Sim-v0'); print(env)"
```

If this fails, call an instructor.

---

## Next Steps (After the Tutorial)

1. **Review the loss:** Open `mile_franka/tutorial/_loss_solution.py` and compare with your implementation.
2. **Read the paper:** https://liralab.usc.edu/mile/ — now that you've *done* MILE, the paper makes sense.
3. **Try your own task:** MILE is task-agnostic. Could you adapt it to your robot?
4. **Contribute:** if you find bugs or improvements, file an issue on [the GitHub repo](https://github.com/rayray2002/mile-franka-tutorial).

---

## FAQ

**Q: Can I run the real-robot queue in parallel with the sim step?**

A: Yes — that's the intended flow. The real-robot station runs while others are still in the sim step.

**Q: What if I want to change the loss after MetaWorld?**

A: Edit `mile_franka/tutorial/loss_exercise.py` and re-run `make tutorial-check-loss`. The next `make tutorial-collect-train` will use your updated loss.

**Q: How many interventions do I need?**

A: 2–3 episodes with 1–2 interventions per episode is enough to see an effect. More is better for convergence, but time is limited.

**Q: What if I run out of time?**

A: The MetaWorld step is the guaranteed "MILE works" demo. The Franka sim improvement is best-effort. If you're short on time, skip eval-after; the before-score still proves the base policy is real.

---

Enjoy the tutorial. You're learning a cutting-edge method for human-in-the-loop robot learning.
