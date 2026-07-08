# Part 4 — MILE on the Real FR3 (sim→real)

**Goal:** run the exact same MILE loop on a live Franka FR3 and feel the sim→real payoff.
Same keyboard, same loss, same policy architecture — only the backend and the cube-pose
source change.

**This runs as a parallel queue.** You don't wait for the whole room. Staff manage a rotation
at the real-robot station while everyone else is in the sim step
([05-franka-sim.md](05-franka-sim.md)).

---

## What is identical vs. what changed

The entire point of Part 4 is how *little* changes from sim to hardware:

| Identical (sim → real) | Changed |
|---|---|
| the Cartesian-impedance controller (same code, same stiffness) | **backend:** multipanda MuJoCo → real FR3 (hucebot controller) |
| **your** `mile_cont_loss_fn` | **pose source:** MuJoCo ground truth → **AprilTag** camera measurements |
| the policy net architecture | teleop device (keyboard, or HTC Vive in the France lab) |
| the keyboard teleop / clutch interface | |

The real cubes are wooden blocks with **AprilTag** markers; `AprilTagPoseSource` (tf2-backed,
live-verified with a D415 camera) replaces the sim's ground-truth poses. A
`TablePlaneCanonicalizer` measures the table height from the resting cube and shifts the
observation into the same frame the policy trained in, so the sim-trained policy sees familiar
numbers.

---

## When it's your turn (staff-run)

**Location:** the real-robot station, staffed by instructors. Staff handle the robot; you do
the teleop.

**Setup (staff):**
1. Confirm the FR3 is powered and homed (it takes ~2 min to boot).
2. Place the tagged cubes on the table.
3. Confirm camera calibration (`MILE_CAMERA_CALIB` / `make calibrate-camera`).

**Commands (staff run these):**
```bash
make franka-up        # launch/verify the FR3 controller stack
make apriltag-up      # camera + apriltag_ros + calibration static tf
make tutorial-collect-train   # you collect interventions with the keyboard
```

**Your experience** is the same clutch-based loop as the sim: watch the policy (ν=0), press
**SPACE** to take over when it's about to fail (ν=1), guide it, release, then **ENTER** to
save the episode. Evaluate before/after with `make eval-real` (staff).

> **Safety:** the policy is never executed autonomously on hardware without a human ready on
> the clutch. `experiment.rollout.auto_eval` is `false` on the real path
> (`config/franka_real.yaml`) precisely so the arm is never driven by an unattended rollout.

---

## The sim→real payoff

Because the controller, loss, and policy architecture are shared, the policy you shaped in
sim transfers to the real arm. The sim taught you *how MILE works*; the real robot shows it
*works on hardware* — with the only real-world additions being vision-based cube tracking and
the physical robot backend. That backend swap (`RobotBackend` ABC) and pose-source swap
(`ObjectPoseSource` ABC) are the two seams the whole `mile_franka` package is designed around.

---

## Wrap-up: what you built

1. **One loss, every task.** You wrote `mile_cont_loss_fn` once. It trained the MetaWorld
   policy (synthetic expert), your Franka sim policy (your keyboard), and the real-robot
   policy. The method is task-agnostic.

2. **Sim→real is one backend swap.** The same impedance controller ran in MuJoCo and on the
   FR3. Only the object-pose source (ground truth → AprilTag) and robot backend changed.

3. **You were the expert.** In Part 1 a synthetic human improved the policy; in Parts 3–4
   *you* did — MILE turned "I see it's about to fail, let me step in" into training signal.

4. **Both signals matter.** MILE learns from your interventions (what to fix) *and* your
   non-interventions (what to leave alone) — the absence of a clutch press is data too.

---

## FAQ

**Why did my Franka policy improve less than MetaWorld?**
MetaWorld uses a synthetic expert with dense data over multiple rounds; your Franka
interventions are sparse (2 episodes, 1 round). MetaWorld is the *guaranteed* demo; Franka is
best-effort. Both are valid MILE.

**Can I run more rounds / collect more?**
Yes — re-run `make tutorial-collect-train`. For MetaWorld you can also continue a prior run
without starting over: `make tutorial-metaworld RESUME=../output_dir` (see
[03-metaworld.md](03-metaworld.md#continue-where-you-left-off-resume)).

**Can I change the loss after Part 1?**
Edit `mile_franka/tutorial/loss_exercise.py`, re-run `make tutorial-check-loss`, and the next
collect-train uses it.

**What if I run out of time?**
The MetaWorld step is the guaranteed "MILE works" result. If short on time, skip the Franka
after-eval; the before-score still proves the base policy is real.

**How would I put this on a different robot?**
Implement the `RobotBackend` ABC and an `ObjectPoseSource` for your hardware. The env, loss,
and training loop are unchanged.

---

## Where to go next

1. **Read the paper:** https://liralab.usc.edu/mile/ — now that you've *done* MILE it will read
   very differently.
2. **Compare your loss** with `mile_franka/tutorial/_loss_solution.py`.
3. **Adapt it to your own task.** MILE is task-agnostic; you'd swap the backend and pose source.
4. **File issues / improvements** on the [repo](https://github.com/rayray2002/mile-franka-tutorial).

Thanks for building MILE end-to-end. Questions? Ask the instructors — see the
[instructor runbook](09-instructor-runbook.md) for the deeper reference.
