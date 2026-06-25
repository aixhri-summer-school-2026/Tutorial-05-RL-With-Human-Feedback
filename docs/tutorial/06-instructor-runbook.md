# MILE Tutorial — Instructor / Staff Runbook

This guide is for the instructors and staff running the 2-hour summer-school tutorial in France. It covers **pre-session prep**, **real-robot safety**, **timing cuts**, **device switching**, and **talking points per tier**.

---

## Pre-Session Prep (France, Outside the 2 Hours)

Complete these tasks **before participants arrive**. Estimated time: 45 minutes + waiting for hardware warmup.

### 1. Mount and calibrate the D415 camera

The real FR3 uses an Intel RealSense D415 to track cube positions via AprilTag.

```bash
# Mount the camera on a bracket or stand pointing at the table
# Serial number: 217222067236 (record for logs)

# In the franka_ros2 container, launch the D415 + AprilTag detector
make apriltag-up

# In another container shell, run the calibration capture
make calibrate-camera

# Expected output:
# "Collecting calibration images: press SPACE to capture, Q to finish"
# Take ~15 snapshots from different angles (include the AprilTag on the board)
# Output: config/camera_calib.yaml
```

**Verify:** `ls -la config/camera_calib.yaml` (should be ~2 KB with extrinsic matrix).

If calibration fails, re-mount the camera and retry. Contact hucebot if the detector can't find tags.

---

### 2. Confirm the hucebot multipanda_ros2 controller is running

The real FR3 is controlled via hucebot's docker stack on **CycloneDDS** (their DDS middleware).

```bash
# On the hucebot controller machine, verify:
docker ps | grep multipanda

# Expected output: container `multipanda_ros2_controller_1` (or similar) is running.

# Test DDS communication from the mile container:
make shell
# Inside the container:
ros2 topic list | grep franka
# Should list /franka_joint_states, /franka_gripper/*, etc.
```

**Verify:** `ros2 topic echo /franka/status` shows live joint state (low latency, <10ms).

If topics are missing, check:
- Both containers on the same docker network (or both on host network + CycloneDDS config).
- The hucebot controller is fully booted (check logs: `docker logs multipanda_ros2_controller_1`).

---

### 3. Test the real FR3 with a smoke test

Before participants touch it, confirm the robot can move.

```bash
make franka-up        # Start the FR3 controller (if not already running)
make franka-shell     # Inside the franka_ros2 container
make real-home-smoke  # Home the robot, move ±5cm in x, then home again

# Expected output:
# "Homing robot…"
# "Testing +X motion…"
# "Testing -X motion…"
# "Returning home…"
# "Smoke test OK"
```

**Verify:** The FR3's visual feedback (arm position) matches the printed coordinates. If it doesn't, contact hucebot support and do **not** proceed.

---

### 4. (Optional) Wire the HTC Vive

The Vive is an optional upgrade for real-robot teleop (more immersive than keyboard). It's not required; keyboard works for everyone.

If you have a Vive:

1. Mount the headset and pair the wand.
2. Verify `ros2 topic list | grep vive` shows hand tracking topics.
3. Record the ROS topic names (handed pose) for the `ViveDevice` instantiation.

**Later, at Tier 4:** If you wire the Vive, swap `"intervener": "keyboard"` → `"intervener": "vive"` in `config/franka_real.yaml`. No other code changes needed.

---

### 5. Tune COST_LOOKUP for the real robot (if needed)

The intervention model uses hyperparameters `[cost, cdf_scale]` per environment. The Franka entry is currently `[2, 2.0]`, tuned for the BC ActorCriticPolicy log-prob scale (much smaller than MetaWorld SAC's `[75–250, 175–200]` range due to the absence of tanh squashing corrections).

**If your real-robot tests show:**
- **Too many interventions (high ν):** increase `cost` (e.g., 2 → 5). The model becomes more selective.
- **Too few interventions (low ν):** decrease `cost` (e.g., 2 → 1). The model asks for help more often.

File: `/home/ray/mile-franka-tutorial/mile/computational_model.py`, line ~50:

```python
COST_LOOKUP = {
    "Franka-Stack-Fake-v0": [2, 2.0],    # Fake backend (training sim)
    "Franka-Stack-Sim-v0": [2, 2.0],     # MuJoCo sim (Tier 3)
    "Franka-Stack-Real-v0": [2, 2.0],    # Real FR3 (Tier 4) — tune if needed
    ...
}
```

Edit the real-robot entry if observed rates diverge from expectations. Commit the change.

---

## Real-FR3 Station Setup & Safety

The real robot is a **shared, staffed resource**. Run the session with **at least one staff member** at the real-robot station at all times.

### Before participants arrive

1. **Power on the FR3.** It takes ~2 minutes to boot.
2. **Mount cubes on the table:** two 5 cm wooden blocks with AprilTag markers glued on (size 0.042 m). Place them side-by-side at the start.
3. **Verify camera calibration:** `ls -la config/camera_calib.yaml`. If missing, run `make calibrate-camera`.
4. **Clear the workspace:** remove obstacles, cables, and tools within 1 m of the table.

### Safety rules (enforce strictly)

1. **Staff drives the robot at all times.**
   - Participants **never** run `make eval-real` or `make mile-real` alone.
   - Participants can run `make tutorial-collect-train`, which only collects data; training loops do not execute the policy autonomously.
   
2. **Keyboard + teleop is always live.**
   - Before handing the keyboard to a participant, test it with a staff member: "Press W; does the arm move forward?" If not, the keyboard might not be connected or focused. Re-check and retry.
   
3. **Emergency stop is within reach.**
   - Mounting: E-stop button on the FR3 pendant or (if available) a software estop in the ROS stack.
   - Protocol: if the arm moves erratically, hit E-stop immediately. Do not try to teleop out of it.
   
4. **Workspace is clear.**
   - Before each group, scan the table for dropped tools, cables, or cubes outside the target zone.
   - If the cubes fall off the table, stop, retrieve them, and power-cycle the FR3 before resuming.

---

### Per-group queue workflow

1. **~5 minutes per group** (orientation + 2–3 episodes).

2. **Participant's flow (staff co-pilots):**
   ```bash
   # Staff runs this; participant watches
   make franka-up      # (if not running)
   make apriltag-up    # (if not running)
   
   # Participant takes the keyboard
   # Participant runs (staff supervises):
   make tutorial-collect-train
   
   # Participant does 2–3 episodes of keyboard teleop
   # (staff watches the arm, ready to E-stop)
   ```

3. **Cleanup:**
   - Participant releases the keyboard.
   - Staff retrieves cubes, resets them to start position.
   - Next participant queues up.

---

## Timing Cuts (If Running Long)

The 2-hour session has built-in slack, but if you're behind, use these cuts in order:

### Cut 1: Drop Tier 2 (saves ~5 minutes)

Tier 2 (fake backend) is a headless sanity check. If time is tight:

1. **Skip** `make tutorial-fake` (Tier 2).
2. **Jump straight** from Tier 1 (MetaWorld) to Tier 3 (MuJoCo sim).

**Trade-off:** participants miss the "same MILE code, new task" moment, but the loss is educational. If any participant doubts the loss is correct, re-run Tier 1 only.

**Expected savings:** ~5 minutes.

---

### Cut 2: Reduce episodes per round (saves ~3–5 minutes)

Edit `config/tutorial_franka.yaml` (Tier 3):

```json
"experiment": {
  "episodes_per_round": 1,  // was 2; reduce to 1
  ...
}
```

This cuts data collection from 2 episodes to 1. The training still runs; participants see loss curves but collect less personal data.

**Trade-off:** fewer tangible interventions, but the principle is the same.

**Expected savings:** ~3–5 minutes (depends on how long participants collect).

---

### Cut 3: Reduce training epochs (saves ~2–3 minutes)

Edit `config/tutorial_franka.yaml`:

```json
"training": {
  "num_epochs": 50,  // was 100; reduce to 50
  ...
}
```

Training still converges; loss curves flatten faster.

**Trade-off:** less refinement, but the policy still improves measurably.

**Expected savings:** ~2–3 minutes.

---

### Cut 4: Tier 4 never gates the room

The real-robot station runs **in parallel**, not sequentially. Tier 4 does **not** delay the end of the session.

- While Tier 3 participants finish, Tier 4 queues rotate through the real robot.
- By 1:45, all Tier 3 participants are done; Tier 4 wraps with the stragglers.
- The 2-hour clock includes Tier 4, but it does not extend the session.

**If you're severely behind on Tier 3 (~15 min late by 1:20):** pause Tier 4 sign-ups and focus on wrapping Tier 3. Re-run Tier 4 next year or in a follow-up session.

---

## Keyboard ↔ Vive Device Switching

### How to switch from keyboard to Vive (real-robot station only)

1. **Confirm Vive is wired and operational:**
   ```bash
   ros2 topic list | grep vive
   # Should list pose topics
   ```

2. **Edit the config:**
   Open `/home/ray/mile-franka-tutorial/config/franka_real.yaml`:
   
   ```json
   {
     "experiment": {
       "intervener": "keyboard",  // ← change to "vive"
       ...
     }
   }
   ```
   
   **Save and commit** (or just edit in-place if not committing).

3. **Restart the container** (if already running):
   ```bash
   make down && make up && make shell
   ```

4. **Run as usual:**
   ```bash
   make franka-up && make apriltag-up
   make tutorial-collect-train
   ```

**That's it.** No code changes, no recompiles. The `TeleopDevice` ABC swaps at runtime.

### Switching back to keyboard

1. **Edit `config/franka_real.yaml`:**
   ```json
   "intervener": "keyboard"
   ```

2. **Restart the container** (optional; config is reloaded on next command).

3. **Run as usual.**

---

## Per-Tier Talking Points

These are the **instructor's verbal anchors** for each tier. Adapt to your audience.

### Tier 1: MetaWorld Peg-Insert (0:00–0:35)

**Framing:**
> "This is MILE on the benchmark from the research paper. Everything is automated — there's no human in the loop, just the downloaded expert and MILE's intervention model. Watch the success rate climb as the policy learns from interventions."

**Key points:**
- The loss you wrote is what trains this policy.
- The synthetic expert (your downloaded model) acts as a ground-truth human.
- Round 0 to Round 1: success improves. That's your loss working.
- By Round N, the policy is learning when the human would help, and it's getting better.

**Success expectation:** success rate increases from Round 0 to Round 1 (exact numbers vary ±0.2 between runs). If success does not increase at all, the loss may have a bug; check the test (`make tutorial-check-loss`).

---

### Tier 2: Franka Fake Backend (0:35–0:42)

**Framing:**
> "This is a 30-second smoke test. It runs one scripted episode on the fake backend — no physics, no ROS — and asserts success. If it prints 'smoke_franka_env ok', the Franka environment pipeline is good and we're clear for the MuJoCo sim."

**Key points:**
- Output is just `smoke_franka_env ok`. No per-episode results — that's expected.
- Obs is 9-dim: `[ee_xyz, gripper_width, top_xyz, bottom_xy]`. Action is 4-DoF: `[Δx, Δy, Δz, gripper_cmd]`.
- If it fails, something is broken in the environment or import chain — fix before Tier 3.

**Optional talking point (if someone asks about observation difference):**
> "MetaWorld obs is 156-dim. Franka obs is 9-dim. Why? Franka cubes don't move in the hand (gripper-centric); we only track 3D position, not full object pose. Fewer dimensions = easier to collect and train on."

---

### Tier 3: Franka MuJoCo Sim with Keyboard Teleop (0:42–1:35)

**Framing (start):**
> "You are the expert now. The policy is running in real-time on the screen. Watch it. When you see it about to fail — say, reaching for empty space or placing the cube too high — hold SPACE and take over. Your actions are recorded. When you release SPACE, the policy continues. After 2–3 episodes, we retrain on your interventions. You'll see the policy improve."

**Key teaching moments:**

1. **Before eval-before:**
   > "We're testing the base policy first so you know where it fails. Record this number — you'll compare it to the after-number."

2. **During collection:**
   > "Remember: you only intervene when the policy is about to fail. Let it succeed on its own as much as possible. Your non-interventions are just as important as your interventions — the model learns that 'when the policy runs freely, the human doesn't step in.'"

3. **After training:**
   > "The loss just trained on your data. The policy and mental model both improve. The mental model learned: 'when this state appears, the human intervenes.'"

4. **After eval-after:**
   > "Your before-score was 0.42. Your after-score is 0.58. You taught the policy 16% better in 3 episodes. That's Tier 3. This is the core of the tutorial."

**Troubleshooting on the fly:**

- **Participant says "the arm didn't move when I pressed SPACE":** Check that the window is focused (click on the MuJoCo window). If still stuck, retry with keyboard keys first (e.g., press W).
- **Participant asks "can I undo a bad intervention?"** No; press BACKSPACE to discard the whole episode and start fresh.
- **Success doesn't improve much (Δ < 0.05):** This is normal if the base policy was already good or the interventions were sparse. Mention that Tier 1's guaranteed improvement is the benchmark; Tier 3 is best-effort.

---

### Tier 4: Real FR3 Block Stacking (1:05–1:45, parallel queue)

**Framing (before queue):**
> "The real robot uses the exact same code as Tier 3. The Cartesian-impedance controller is identical. The only differences: the cubes are tracked by AprilTag vision instead of simulation ground-truth, and the robot is real FR3 hardware instead of MuJoCo. Everything else is the same — your loss, the policy net, the MILE loop. You'll use the same keyboard to teleop. This is sim-to-real transfer."

**As participants rotate in:**
> "When it's your turn, we'll have the real robot running, cubes positioned, and the system ready. You'll collect 2–3 episodes, same as Tier 3. Staff will watch the arm and e-stop if anything goes wrong. Your interventions train a new policy on real-world data. If we have time, we'll eval the new policy on the real robot."

**Safety reminder (before handing keyboard):**
> "This is real hardware. If the arm moves weirdly or you lose control, release the keyboard immediately. Staff are watching with an E-stop. The worst case is a slow motion; we'll recover. Do not try to correct it with more keyboard input — let staff handle it."

**Talking point (during/after):**
> "Notice anything different? The latency is a bit higher, the gripper is actually flexible, the cubes are real. But the policy is running the same algorithm. When you intervened with the keyboard, you were teaching the real robot, not a sim."

**Wrap-up (if time allows):**
> "Your sim policy improved by 16% on your interventions. Can a real-robot policy do the same? We don't have time to fully retrain, but you just collected real data. This is the seed for the next iteration."

---

### Wrap-Up (1:45–2:00)

**Main talking points:**

1. **One loss, four tasks:** "You implemented `mile_cont_loss_fn` once. That loss trained MetaWorld (Tier 1), the Franka sim (Tier 3), and the real robot (Tier 4). MILE is task-agnostic."

2. **Sim→real is one backend swap:** "The Franka sim and real robot use the same Cartesian-impedance controller. The only difference: sim observations come from MuJoCo ground truth; real observations come from AprilTag markers. Everything else — the policy, the intervention model, the loss — is identical."

3. **You are the expert:** "In Tier 1, a synthetic expert (downloaded model) improved the policy. In Tiers 3 and 4, you were the expert. MILE learns from human judgment. When you see the robot about to fail, you step in. That signal is the training data."

4. **Joint training matters:** "The policy learns the right action. The mental model learns when you'd intervene. They train together. By the end, the intervention model can predict: 'the human will step in here.' The policy can then avoid those states."

---

## Key Config Files for Instructors

Quickly reference these files if you need to adjust things on the fly:

1. **`config/tutorial_franka.yaml`** (Tier 3, sim):
   - `episodes_per_round`, `num_epochs`: reduce if running long.
   - `intervener`: `keyboard` (default) or `joystick` or `vive` (if wired).

2. **`config/franka_real.yaml`** (Tier 4, real):
   - `intervener`: same as above. Swap to `vive` if you wire the headset.
   - `rollout.auto_eval`: set to `false` (never auto-eval the policy on real hardware).

3. **`mile/computational_model.py`** (loss hyperparams):
   - Line ~50: `COST_LOOKUP["Franka-Stack-Real-v0"]`: tune if real-robot intervention rate is off.

4. **`config/camera_calib.yaml`** (real-robot extrinsics):
   - Verify this exists before Tier 4. If missing, run `make calibrate-camera`.

---

## Common Staff Questions

**Q: What if a participant's loss test fails?**

A: Show them `mile_franka/tutorial/_loss_solution.py` (the answer key). Have them compare line by line. Common bugs: forgot to clamp probabilities before log, didn't mask ν=1 steps, wrong weight balance. After the fix, re-run `make tutorial-check-loss`.

**Q: What if the real robot doesn't respond to the keyboard?**

A: Check:
1. Is the keyboard plugged in and focused on the terminal window? (Click the terminal, try pressing a key.)
2. Is the FR3 powered and homed? (`ros2 topic echo /franka/status`)
3. Is the MuJoCo sim running? (`make sim-up` status)
If all are OK, restart the franka_ros2 container: `make franka-down && make franka-up`.

**Q: How much should a participant intervene?**

A: 1–2 interventions per episode is enough. They don't need to over-engineer. The policy is already mediocre (0.4 success); they just need to show it 2–3 failure modes to improve.

**Q: Can Tier 4 (real robot) run in parallel with Tier 3 (sim)?**

A: Yes, and it should. Set up a separate queue station. While one group is collecting in sim, another is collecting on the real robot. Tier 4 never gates the 2-hour clock.

**Q: What if someone wants to run more than 2–3 episodes?**

A: Allowed, but note that each episode is ~1 minute of real time (teleop + data save). A 5-episode session takes 5–7 minutes. If they're having fun and time allows, let them. If the clock is tight, politely cut after 3.

**Q: Can the mental model be wrong?**

A: It's learned from the data, so it's only as good as the interventions. If a participant's interventions are inconsistent (e.g., sometimes intervening on the same state, sometimes not), the mental model will have high loss but still train. It's normal.

---

## Post-Session Checklist

After the 2-hour session:

1. **Power down the FR3.** (Homing not required; it can be powered off as-is.)
2. **Retrieve the cubes** and store them safely.
3. **Shut down the containers:**
   ```bash
   make down
   ```
4. **Collect participant feedback** (optional survey form or informal chat).
5. **Review any errors** in the logs:
   ```bash
   docker logs mile_sim 2>&1 | tail -50
   ```
6. **Archive participant data** (if collecting for research):
   ```bash
   ls -la output_dir/franka/
   ```
7. **Note any hardware issues** for the next session (broken camera, wobbly robot, etc.).

---

## FAQs for Instructors

**Q: Should I tell participants the answer key exists?**

A: No. They should implement the loss themselves. If they get stuck (>15 min), give them a hint: "Check that ν=1 steps use Gaussian NLL, and ν=0 steps use only BCE. Are you masking correctly?" If they still fail, mention the answer key and let them review it later.

**Q: Can I skip MetaWorld (Tier 1)?**

A: Not recommended. Tier 1 is the **guaranteed improvement demo**. It validates the loss in a clean setting. Without it, if Tier 3 doesn't improve much, participants might doubt the loss. Keep Tier 1.

**Q: What if the D415 camera drifts during Tier 4?**

A: AprilTag markers can be noisy over long sessions. If cube tracking degrades (position jumps, disappears), recalibrate:
```bash
# Quick recalibration (5 min)
make calibrate-camera
```
Or just remind participants: "The vision is noisy; take that into account when steering."

**Q: Can I run the tutorial without Tier 4 (no real FR3)?**

A: Yes. Tiers 1–3 are complete on a laptop. Tier 4 is the real-hardware payoff; if it's not available, the tutorial still teaches MILE end-to-end in sim. Note this in the opening remarks.

---

## Contacts & Resources

- **hucebot controller support:** [hucebot GitHub issues](https://github.com/hucebot/multipanda_ros2)
- **MILE paper & code:** https://liralab.usc.edu/mile/
- **Tutorial repo issues:** [GitHub issues](https://github.com/rayray2002/mile-franka-tutorial)
- **This runbook:** `/home/ray/mile-franka-tutorial/docs/tutorial/06-instructor-runbook.md`

---

**Good luck. Enjoy running MILE at the summer school. The participants are about to learn something cool.**
