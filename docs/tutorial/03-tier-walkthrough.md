# MILE Tutorial — The 2-Hour Hands-On Flow

This is your step-by-step guide for the live 2-hour tutorial. **You've already completed the setup** (Sections 0–2). Now it's time to run the full MILE loop: from a synthetic expert (Tier 1) through your own interventions (Tier 3) to the real robot (Tier 4).

**All commands run inside the container.** To start:
```bash
make up && make shell
```

You're now inside the `mile` container with the environment sourced. All timestamps below assume you start the clock at 0:00 when you open this document.

---

## 0:00–0:10 · Land & Verify (10 min)

**Goal:** Confirm your setup is ready; get a 5-min mental model of MILE.

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

### Step 2: Read the concepts (5 min)

While your brain settles, read **[01-concepts.md](01-concepts.md)** — it's the anchor for everything you're about to do:

- **ν (nu):** the binary "did the human intervene?" flag on each timestep.
- **The intervention model p(ν=1|s):** the robot's prediction of when you'll step in.
- **Joint training:** the policy (what the robot executes) and the mental model (what you think it will do) train together.
- **The loss you'll write:** BCE on ν + Gaussian NLL on ν=1 steps.
- **The 4-tier ladder:** MetaWorld → Franka fake → Franka sim → real FR3.

You're now ready to implement MILE's heart.

---

## 0:10–0:22 · Implement the Loss (12 min)

**Goal:** Write `mile_cont_loss_fn` — the loss used by all four tiers.

**File to edit:** (from inside the container)
```bash
nano mile_franka/tutorial/loss_exercise.py
```

Or open it in your favorite editor on the host and it'll sync into the container.

**What to do:**

Read the docstring and the **[02-loss-exercise.md](02-loss-exercise.md)** handout. You'll implement:

1. **Discrete loss** (BCE): does the model predict the right intervention flag?
2. **Continuous loss** (Gaussian NLL): when ν=1, does the model predict the human's action?
3. **Combine them** with weights.

The handout includes the math, tensor shapes, and a worked solution for reference.

**Test your implementation:**

```bash
make tutorial-check-loss
```

**Expected output (if correct):**
```
test_loss_exercise.py::test_solution_matches_itself_known_values PASSED
test_loss_exercise.py::test_candidate_matches_reference_when_implemented PASSED
test_loss_exercise.py::test_continuous_term_only_uses_intervention_steps PASSED

=== 3 passed ===
```

**If tests fail:** read the error, fix the bug, and re-run. Common issues:
- Forgot to clamp the probabilities before taking log?
- Masking only ν=1 steps for the continuous loss?
- Combining with the right weights?

**Stuck or out of time?**

No worries — the answer key will auto-apply when you train, and you can review the solution in `mile_franka/tutorial/_loss_solution.py` later. Move on.

---

## 0:22–0:40 · Tier 1: MetaWorld (Your Loss on the Paper Benchmark)

**Goal:** Watch MILE train on the classic peg-insertion task with the loss you just wrote.

**Setup:**

No human involved — you're watching an *automated synthetic expert* intervene. The point is to validate that your loss works on clean sim signal.

**Run Tier 1:**

```bash
make tutorial-metaworld
```

**What to expect:**

The script will print:
```
Starting MetaWorld peg-insert benchmark (synthetic human)
Round 0: collecting…
  [done]
  training (epoch 1/100)…
  success_rate: 0.18 → 0.35
Round 1: collecting…
  [done]
  training (epoch 1/100)…
  success_rate: 0.35 → 0.52
Final success rate: 0.52
```

**What's happening:**

1. Round 0: the synthetic human (your downloaded expert + MILE's intervention model) collects data as the initial policy tries and fails.
2. The policy trains on interventions — your loss guides it.
3. Round 1: the improved policy collects data. Success climbs (0.35 → 0.52).
4. Each round, the policy gets better at the peg-insertion task.

**Duration:** ~10 min (mostly waiting for training).

**The key insight:** "Success improved from Round 0 to Round 1. I wrote the loss. My loss made this happen."

This is your proof that the loss is correct *and* that MILE works. In Tier 3, you'll replace the synthetic expert with your own keyboard interventions.

---

## 0:40–0:45 · Tier 2: Franka Fake Backend (Headless, Quick)

**Goal:** See the Franka task on the simplest backend (no physics, no ROS).

**Run Tier 2:**

```bash
make tutorial-fake
```

**Expected output:**
```
Evaluating mediocre base policy on Franka-Stack-Fake-v0…
Episode 1: success=False (gripper missed the cube)
Episode 2: success=True
Episode 3: success=False (stacked, but fell on the second cube)
…
Mean success rate: 0.4
```

**What's happening:**

- The mediocre **base policy** (trained offline from sim rollouts) runs on the **fake backend** — a simple kinematic world with no MuJoCo physics.
- It succeeds ~40% of the time. It fails in predictable ways: misses the cube, stacks but wobbles, places too high.
- This is where MILE will help: your interventions (Tier 3) will teach it to handle these failures.

**Why this tier?**

It's a 5-minute sanity check. The policy runs at full speed (no graphics), and you see the observation/action dimensions:
- **Obs:** 9-dim (EE position, gripper width, cube position) × `FrameStack(10)` = 90-dim.
- **Action:** 4-DoF (Δx, Δy, Δz, gripper command).

**Duration:** ~2 min.

**Fallback for slow laptops:** if your MuJoCo rendering is sluggish in the next step, you can re-run this to see observations without graphics overhead.

---

## 0:45–1:00 · Visual Sim & Learn the Keyboard (15 min)

**Goal:** Start the MuJoCo sim and practice the teleop controls.

### Step 1: Launch the headless simulator

In the container, start the simulation in the background:

```bash
make sim-up
```

**Expected output:**
```
sim launching headless; give it ~10s, then check: make collect-mediocre
```

Wait ~10 seconds for the sim to boot.

### Step 2: Open the viewer on the host

On your **host machine** (laptop, not inside the container), open a terminal and run:

```bash
make sim-gui
```

**Expected output:**

A MuJoCo window opens. You'll see:
- The Franka robot (light blue arm) at home.
- Two red cubes on the table (the stack target).
- A 3D viewport.

**Note on rendering:** If the window is slow or doesn't appear, see [05-troubleshooting.md](05-troubleshooting.md#gl-rendering-slow). You can skip the visual and proceed headless to Tier 3 using `make eval-base` and `make tutorial-collect-train` without `sim-gui`.

### Step 3: Learn the keyboard (free-play teleop)

**Inside the container**, open a shell (or use your existing one) and run:

```bash
make tutorial-teleop
```

This launches a free-play mode where you can experiment with the keyboard. Refer to **[04-teleop.md](04-teleop.md)** for the full key map, but here's the cheat sheet:

| Key | Action |
|---|---|
| `w` / `s` | move forward/backward (Δy) |
| `a` / `d` | move left/right (Δx) |
| `q` / `e` | move up/down (Δz) |
| **SPACE** | **toggle intervention (ν)** — hold to take over |
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

- When **ν=0** (no clutch), the policy is commanding the arm. You see the policy's motion in real-time.
- When **ν=1** (clutch held), **you** command the arm. The policy learns from this.
- In Tier 3, you'll hold the clutch *only when the policy is about to fail*, teaching it to correct itself.

**Duration:** 5–10 min of free-play to get comfortable. When you feel confident, move to Tier 3.

---

## 1:00–1:30 · Tier 3: Be the Human in Sim (30 min)

**Goal:** Collect your own interventions, train on your data, watch your policy improve.

**This is the core experience:** you replace the synthetic expert. Your interventions teach your own policy.

### Step 1: Evaluate the base policy (before)

Stop the free-play teleop. In the container, run:

```bash
make eval-base
```

**Expected output:**
```
Evaluating base policy (10 episodes)…
Episode 1: success=True
Episode 2: success=False
…
Mean success rate: 0.42
Std: 0.10
Videos saved to: output_dir/franka/eval_videos_TIMESTAMP/
```

**Record the before-score.** In this example, it's **0.42** (42% success).

### Step 2: Collect interventions with keyboard teleop

Run the combined collect-and-train command:

```bash
make tutorial-collect-train
```

**What happens:**

1. The container launches the training system.
2. The sim displays the arm. The policy is running.
3. **You watch and intervene:** when you see the policy is about to fail (reaching for empty space, gripper off-target, etc.), **hold SPACE** and take over.
4. Once you fix the situation, **release SPACE** and let the policy continue.
5. When the episode ends (cube stacked or failed), press **ENTER** to mark the episode done. Press **BACKSPACE** to discard if you want to retry.

**How much to collect:**

- Aim for **2–3 episodes** with 1–2 interventions per episode.
- Don't over-engineer — "good enough" is fine. You're teaching the policy to handle realistic human feedback.
- The policy updates immediately after training, so you'll see the effect.

**Expected timeline:**

```
Episode 1: collecting with keyboard… (30–60 sec of gameplay)
  interventions: 2
  [saving]
Episode 2: collecting with keyboard…
  interventions: 1
  [saving]

Training on 2 episodes (3 interventions total)…
Round 0 complete; retraining policy...
  policy loss: 0.23
  mental_model loss: 0.12
  [done]
```

**What you're training:**

The same loss you implemented in Tier 1, now on *your* data. The policy learns from your interventions (ν=1) and respects your non-interventions (ν=0).

### Step 3: Evaluate the trained policy (after)

After training finishes, run:

```bash
make eval-base
```

**Expected output:**
```
Evaluating trained policy (10 episodes)…
…
Mean success rate: 0.58
Std: 0.11
Videos saved to: output_dir/franka/eval_mile_videos_TIMESTAMP/
```

**Record the after-score.** In this example, it's **0.58**.

**The delta:** 0.42 → 0.58 = **+0.16 (16% improvement)**.

**Interpret the result:**

- **Visible improvement** (Δ > 0.10): great! Your interventions directly taught the policy.
- **Small improvement or no change** (Δ ≤ 0.05): the base policy was already close to optimal, or your interventions didn't cover the failure mode. This is normal from sparse data; the *guaranteed* improvement demo is Tier 1 (MetaWorld), which removes the pressure on Franka's sparse interventions.

**Review the videos:**

In the output directory, you'll find videos of successful episodes. Watch them — you'll see the policy executing the corrections you taught it.

### Duration

This step takes **20–30 min** total:
- 5–10 min: eval-before.
- 10–15 min: collecting with teleop + training.
- 5 min: eval-after + reflection.

---

## ~1:05–1:45 · Tier 4: Real FR3 (Parallel Queue)

**Goal:** Rotate through the live robot and experience sim→real transfer.

**This happens in parallel with Tier 3.** You don't wait for the entire room to finish sim work. Instead, staff manage a queue at the real-robot station.

### When it's your turn:

**Location:** the real-robot station (staffed by instructors).

**Setup (staff will handle):**

1. Confirm the FR3 is powered and homed.
2. Mount cubes on the table.
3. Confirm camera calibration (`MILE_CAMERA_CALIB`).

**Your experience:**

- **Same keyboard** (or HTC Vive if wired) as Tier 3.
- **Same policy** trained in your laptop's sim.
- **Same MILE loop** — collect interventions, retrain, eval.

**Commands (staff runs these):**

```bash
# Launch the FR3 controller (staff)
make franka-up

# In parallel, staff launch camera + AprilTag (staff)
make apriltag-up

# You collect interventions with keyboard (or Vive)
make tutorial-collect-train
```

**The cubes:**

Real wooden cubes are tracked via AprilTag markers (vision) instead of MuJoCo ground truth. Everything else is identical to Tier 3.

**Expected timeline:**

- **5 min per participant** (load time + 2–3 episodes).
- **Parallel queue:** while you're collecting, the next person starts eval-before on the sim.
- **No waiting:** Tier 4 doesn't gate the room; the clock doesn't stop.

**What's identical (sim→real):**

- The Cartesian-impedance controller (same code, same stiffness).
- The MILE loss (your implementation).
- The policy net architecture.
- The keyboard teleop interface.

**What changed:**

- **Backend:** multipanda ROS2 sim → real FR3 (hucebot controller).
- **Pose source:** MuJoCo ground truth → AprilTag camera measurements.
- **(Optional) Teleop device:** keyboard → HTC Vive (if wired).

This is the **sim→real transfer payoff**: the sim taught you how MILE works; the real robot proves it works on hardware.

---

## 1:45–2:00 · Wrap & Q&A (15 min)

**Goal:** Reflect on what you built and ask questions.

**Key takeaways:**

1. **One loss, four tasks:** you wrote `mile_cont_loss_fn` once, and it trained the MetaWorld policy (Tier 1), the Franka sim policy (Tier 3), and the real-robot policy (Tier 4). That's the power of MILE — the method is task-agnostic.

2. **Sim→real is one controller swap:** the same robot controller ran in sim and on hardware. The only differences were the observations (sim ground-truth vs. real AprilTag) and (optionally) the teleop device.

3. **You are the expert:** in Tier 1, a synthetic expert improved the policy. In Tier 3 and 4, *you* were the expert. MILE learns from human intuition — when you see the policy about to fail, you step in.

4. **Joint training matters:** the policy and mental model train together. Your interventions teach both: the policy learns the right action, and the mental model learns when you'd intervene.

**Talking points (instructor will cover):**

- Why the intervention flag ν is central (not just mimicry, but *when* to ask for help).
- How the probit intervention model decides when p(ν=1|s) is high enough to ask.
- Why Tier 1's guaranteed improvement (MetaWorld) is separate from Tier 3's best-effort (sparse Franka data).
- The real-robot transfer: what worked, what surprised you, what failed.

**Questions?**

Ask the instructors. They've run this loop many times. Common questions:

- "Why did my policy improve less than Tier 1?" → Tier 1 uses a synthetic expert with dense data; your Franka interventions are sparse. Both are valid; Tier 1 is the guarantee.
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

If still stuck, fall back to headless (skip sim-gui, proceed to Tier 3 directly).

### Keyboard doesn't respond in sim-gui

Press Escape to focus the window, then try again. If the sim is running (verify with `ros2 topic list | grep sim`), the teleop should work.

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

**Q: Can I run multiple tiers in parallel?**

A: Yes. Tier 4 (real FR3) runs in parallel with Tier 3. Tiers 1–2 are faster and run first.

**Q: What if I want to change the loss after Tier 1?**

A: You can edit `mile_franka/tutorial/loss_exercise.py` and re-run `make tutorial-check-loss`. The next `make tutorial-collect-train` will use your updated loss. (Tier 1 won't re-run unless you manually delete the logs.)

**Q: How many interventions do I need?**

A: 2–3 episodes with 1–2 interventions per episode is enough to see an effect. More is better for convergence, but time is limited.

**Q: What if I run out of time?**

A: Tier 1 is the guaranteed "MILE works" demo. Tier 3's improvement is best-effort. If you're short on time, skip Tier 3's eval-after; the before-score still proves the base policy is real.

---

Enjoy the tutorial. You're learning a cutting-edge method for human-in-the-loop robot learning. 🤖
