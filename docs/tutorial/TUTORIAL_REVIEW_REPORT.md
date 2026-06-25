# MILE Tutorial Review Report

**Reviewer:** Claude Code  
**Date:** 2026-06-24  
**Method:** Followed the tutorial docs step-by-step as a student would, running all commands inside the Docker container (`make shell` / docker exec pattern).

---

## Setup Verification (00-setup.md)

`make tutorial-check` was run inside the container. Result:

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

**Status: PASS.** All imports and artifacts present.

---

## Loss Exercise (02-loss-exercise.md)

### Before implementation — `make tutorial-check-loss`

**BUG FOUND AND FIXED:** Running `make tutorial-check-loss` from the host (or from inside `make shell`) failed due to a pytest/ROS plugin conflict:

```
pluggy._manager.PluginValidationError: Plugin 'launch_testing' for hook 'pytest_pycollect_makemodule'
```

Root cause: `in_container_env.sh` sources ROS (`/opt/ros/humble/setup.bash`), which injects the `launch_testing_ros` pytest plugin into the Python path. This plugin is incompatible with pytest 9.1.0 (installed in the container).

**Fix applied:** Changed `tutorial-check-loss` in the Makefile from a host-only `python3 -m pytest` call to `$(DC) exec sim bash -c '...'` that runs without sourcing the ROS environment. This makes it work from both the host and from inside `make shell`.

**Before fix result (from inside container shell):**
```
PluginValidationError: unknown hook 'pytest_launch_collect_makemodule'
```

**After fix — scaffold state (expected: SKIP):**
```
tests/test_loss_exercise.py::test_solution_matches_itself_known_values PASSED
tests/test_loss_exercise.py::test_candidate_matches_reference_when_implemented SKIPPED
tests/test_loss_exercise.py::test_continuous_term_only_uses_intervention_steps PASSED
2 passed, 1 skipped
```

### After implementing the loss

Implemented `mile_cont_loss_fn` per the exercise docstring:

```
tests/test_loss_exercise.py::test_solution_matches_itself_known_values PASSED
tests/test_loss_exercise.py::test_candidate_matches_reference_when_implemented PASSED
tests/test_loss_exercise.py::test_continuous_term_only_uses_intervention_steps PASSED
3 passed in 1.05s
```

**Status: PASS** (after Makefile fix and correct implementation).

---

## Tier 1: MetaWorld (tutorial-metaworld)

### BUG FOUND AND FIXED — deepcopy crash

Running `make tutorial-metaworld` (from inside the container) crashed at the end of Round 0 training with:

```
RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the
deepcopy protocol at the moment. If you were attempting to deepcopy a module, this may
be because of a torch.nn.utils.weight_norm usage, see pytorch/pytorch#103001
```

**Location:** `mile/algorithm.py:347` — `best_mental_model = deepcopy(self.mental_model)`

**Root cause:** The SAC policy uses `weight_norm` internally. PyTorch's `deepcopy` cannot clone weight_norm tensors because they are non-leaf nodes in the autograd graph.

**Fix applied in `mile/algorithm.py`:** Replaced `deepcopy(policy)` / `deepcopy(mental_model)` with save/load checkpointing using the models' own `.save()` method. The best checkpoint is saved to temp files (`_best_policy_ckpt`, `_best_mental_model_ckpt`) and copied to the final destination only if `on_best_rollout_success_rate: true`.

### Results after fix

```
Round 0:
  Initial success rate (before training): 0.1 (10%)
  Dataset size collected: 500 samples
  Percentage no-intervention: 59.6%
  Training: 50 epochs, cont_loss 29.8→2.29, disc_loss 0.68→0.678
  Post-Round-0 eval: success_rate = 0.1

Round 1:
  Training: 50 epochs on combined dataset
  Post-Round-1 eval: success_rate = 0.3 (30%), init_success_rate = 0.2
```

**Delta: 0.1 → 0.3 = +20% success rate improvement.**

The policy saved to `output_dir/policy`, `output_dir/mental_model`.

**Status: PASS** (after deepcopy fix in algorithm.py). Training ran ~4 minutes.

**Doc discrepancy:** The tutorial docs show expected output:
```
success_rate: 0.18 → 0.35
Final success rate: 0.52
```
Actual observed: 0.1 → 0.3. This is within normal variance — the docs' example was illustrative, not guaranteed. No fix needed, but a note in the docs that numbers vary would help participants manage expectations.

---

## Tier 2: Fake Backend (tutorial-fake)

### BUG FOUND — Output completely different from docs

Running `make tutorial-fake` (which calls `python3 scripts/smoke_franka_env.py`):

**Actual output:**
```
smoke_franka_env ok
```

**Expected output per docs:**
```
Evaluating mediocre base policy on Franka-Stack-Fake-v0…
Episode 1: success=False (gripper missed the cube)
Episode 2: success=True
…
Mean success rate: 0.4
```

**Root cause:** `smoke_franka_env.py` is a unit test that runs one **scripted** (not mediocre) episode on the fake backend and asserts success with `assert`. It does NOT:
- Use the mediocre base policy from `trained_models/franka/base_policy`
- Run multiple episodes
- Print per-episode results or compute a success rate

**Additional finding:** Running `eval_base_policy_sim.py --env_name Franka-Stack-Fake-v0 --episodes 5` (what the docs appear to describe) gives:
```
episode 0: success=0 steps=150
episode 1: success=0 steps=150
...
BASE POLICY SIM SUCCESS RATE = 0.00 (0/5); mean steps=150
```
The base policy gets 0% on the fake backend in 150 steps max — the fake backend's `max_steps=150` is too short for the policy to succeed, and the policy trained for the sim backend may not transfer to the fake world.

**Recommendation:** The simplest fix is to update the tutorial docs to describe Tier 2 honestly as a "smoke test that confirms the Franka environment loads correctly." Changing the `tutorial-fake` target to run a multi-episode base policy eval would need either a longer `max_steps` for the fake backend or a different policy. This is a design decision for the course authors.

---

## Tier 3: Franka Sim — eval-base

### Result

`make sim-up` started the MuJoCo simulator. After waiting ~15 seconds for boot, the ROS service `/get_body_state` was confirmed available.

`make eval-base` was run (5 episodes, using `--episodes 5`). Results:
```
episode 0: success=0 steps=1000
episode 1: success=1 steps=138
episode 2: success=0 steps=1000
episode 3: success=1 steps=97
episode 4: success=1 steps=688

BASE POLICY SIM SUCCESS RATE = 0.60 (3/5); mean steps=585
```

**Before-training baseline: 60% success rate (3/5 episodes).**

Note: success is stochastic (policy uses `deterministic=False`). Expect roughly 40–70% depending on the random draw. The docs' example of 42% is plausible but not guaranteed.

**Sim stability note:** After a Ctrl-C kill of the eval, the sim arm may not settle properly on the next reset (`home did not settle within 4.0s: error 2.2cm`). Always do `make sim-up` to restart the sim headless process before running eval if the previous run was interrupted.

### Timing concern for tutorial

The `max_steps=1000` at 10 Hz means a failing episode takes 100 seconds. Ten episodes = up to 16 minutes. The tutorial allocates 5 minutes for eval-before. **Recommendation:** Reduce to `--episodes 5` in the Makefile's `eval-base` target, saving 8+ minutes.

**Actual output format vs. docs:**

The docs describe the output as:
```
Evaluating base policy (10 episodes)…
Episode 1: success=True
Mean success rate: 0.42
Std: 0.10
Videos saved to: output_dir/franka/eval_videos_TIMESTAMP/
```

The actual `eval_base_policy_sim.py` outputs:
```
episode 0: success=1 steps=135
...
BASE POLICY SIM SUCCESS RATE = 0.40 (2/5); mean steps=647
```

Differences:
- No "Evaluating base policy..." header
- Episodes are 0-indexed, not 1-indexed
- `success=0/1` (int), not `success=True/False` (bool)
- No std deviation printed
- No "Videos saved to:" summary line
- Actual header uses all-caps: `BASE POLICY SIM SUCCESS RATE`

---

## Bugs Fixed Summary

| # | File | Bug | Fix Applied |
|---|------|-----|-------------|
| 1 | `Makefile` | `tutorial-check-loss` runs host `python3 -m pytest` — fails due to ROS pytest plugin conflict when called from `make shell` | Changed to `$(DC) exec sim bash -c` without ROS env sourced |
| 2 | `Makefile` | `tutorial-teleop` target missing — referenced in docs but undefined | Added target + created `scripts/tutorial_teleop.py` |
| 3 | `Makefile` | `tutorial-check`, `tutorial-check-loss`, `tutorial-metaworld`, `tutorial-fake`, `tutorial-teleop` were not in `.PHONY` | Added all tutorial targets to `.PHONY` |
| 4 | `mile/algorithm.py` | `deepcopy(policy)` / `deepcopy(mental_model)` crashes with PyTorch weight_norm | Replaced with `.save()` / file copy checkpointing |
| 5 | `docs/tutorial/03-tier-walkthrough.md` | After-training eval command was `make eval-base` but should be `make eval-mile` | Fixed to `make eval-mile` |
| 6 | `docs/tutorial/03-tier-walkthrough.md` | Keyboard axis labels wrong: W/S labeled Δy, A/D labeled Δx | Fixed to match code: W/S=Δx, A/D=Δy |
| 7 | `docs/tutorial/03-tier-walkthrough.md` | SPACE described as "hold to take over / release to stop" | Fixed to describe toggle behavior (code uses `segment_mode=True`) |
| 8 | `docs/tutorial/04-teleop.md` | A = Move left (-Y), D = Move right (+Y) — signs inverted vs. code | Fixed to A=+Y, D=-Y per `keyboard.py` |
| 9 | `docs/tutorial/04-teleop.md` | `lsjs /dev/input/js*` — `lsjs` is not a real command | Fixed to `ls /dev/input/js*` |
| 10 | `docs/tutorial/06-instructor-runbook.md` | COST_LOOKUP shown as `[70, 100.0]` — actual code has `[2, 2.0]` | Updated to match `mile/computational_model.py` |
| 11 | `docs/tutorial/06-instructor-runbook.md` | References `config/tutorial_franka_real.yaml` (does not exist) | Fixed to `config/franka_real.yaml` (4 occurrences) |

---

## Feedback: Improvements to Tutorial Code and Flow

### High Priority

**F1. Tier 2 output is misleading**  
`make tutorial-fake` should show a multi-episode mediocre base policy evaluation (as the docs describe), not just a smoke test. Either fix the script or change the docs to match. Participants expecting per-episode results get "smoke_franka_env ok" with no data.

**F2. eval-base and eval-mile are too slow for a 2-hour tutorial**  
Each failed episode runs for 100 seconds (1000 steps × 0.1s). The tutorial allocates 5–10 minutes for eval, but 10 episodes can take 16+ minutes. Recommended fix: add `--episodes 5` to `make eval-base` and `make eval-mile` in the Makefile, reducing wait time to ≤8 minutes.

**F3. MetaWorld Tier 1 output is too verbose**  
The training loop prints 50 log lines per epoch (batch + epoch table), totaling 100+ lines per round × 2 rounds = 200+ lines. Participants can't easily see the success rate trend. Recommended: add a compact summary line like `Round 0 → Round 1: success_rate 0.10 → 0.30` at the end of training.

**F4. tutorial-metaworld / tutorial-collect-train confuse host vs. container context**  
These targets run as bare shell commands (no `$(call RUN,...)`), so they only work from inside `make shell`. A student who tries `make tutorial-metaworld` on the host sees `ModuleNotFoundError: No module named 'torch'` — a confusing error. Either add these to the `$(call RUN,...)` pattern or add a guard that prints "run this from inside make shell" if torch isn't available.

### Medium Priority

**F5. eval-base vs. eval-mile naming is ambiguous for students**  
Students must remember to run `eval-base` before training and `eval-mile` after. A clearer naming like `eval-before` / `eval-after`, or a single `make eval` that detects which policy to use, would reduce confusion.

**F6. Actual eval output format doesn't match docs**  
The expected output in `03-tier-walkthrough.md` shows:
```
Evaluating base policy (10 episodes)…
Episode 1: success=True
Mean success rate: 0.42
Std: 0.10
Videos saved to: ...
```
Actual output:
```
episode 0: success=1 steps=135
BASE POLICY SIM SUCCESS RATE = 0.XX (N/10); mean steps=Y
```
Update the docs to match the real output format (or improve the script).

**F7. No feedback when SPACE toggle is active**  
Inside `tutorial-collect-train`, when a student presses SPACE to enter ν=1, there's no immediate terminal confirmation (the segment ON/OFF messages print a line, but only when the state changes). A persistent status indicator (e.g., `[ν=1 INTERVENTION]` shown in the terminal each step) would help students understand when they're controlling the robot.

**F8. No `tutorial_franka_real.yaml` exists**  
The instructor runbook repeatedly references this file (now fixed to `franka_real.yaml`). A dedicated tutorial-safe real config (`config/tutorial_franka_real.yaml`) with `rollout.auto_eval: false` and safe defaults should be created to prevent instructors from accidentally running auto-eval on the real robot.

### Low Priority

**F9. `02-loss-exercise.md` BCE vs. NLL terminology mismatch**  
The concepts doc (01) calls the intervention loss "BCE (binary cross-entropy)". The exercise doc (02) calls it "NLL (Negative Log Likelihood)" computed via `F.nll_loss`. These are equivalent but the terminology difference (and the use of a 2-class softmax rather than sigmoid) may confuse participants expecting a `BCELoss` call. A note that `F.nll_loss(log_softmax_probs, labels)` is the multi-class generalization of BCE would clarify this.

**F10. Circular reference between 03 and 04**  
`03-tier-walkthrough.md` sends students to `04-teleop.md` for the key map. `04-teleop.md`'s "Next steps" section sends them back to `03-tier-walkthrough.md`. The cross-reference is confusing when read linearly. Make `04-teleop.md` a pure reference document with no forward links to `03`.

**F11. Success rate variance is not set up well in the docs**  
The MetaWorld example shows specific numbers (0.18→0.35→0.52). The actual numbers depend on random seeds and hardware. Participants who see 0.1→0.3 may think something is wrong. A note that "numbers vary by ±0.2 between runs" would prevent unnecessary debugging.

---

## Files Changed

| File | Change |
|------|--------|
| `Makefile` | Fixed `tutorial-check-loss` to use container exec; added `tutorial-teleop`; expanded `.PHONY` |
| `mile/algorithm.py` | Fixed `deepcopy` crash with save-based checkpointing |
| `mile_franka/tutorial/loss_exercise.py` | Implemented the loss (student step, not a bug fix) |
| `scripts/tutorial_teleop.py` | Created new free-play teleop script |
| `docs/tutorial/03-tier-walkthrough.md` | Fixed eval command, axis labels, SPACE toggle description |
| `docs/tutorial/04-teleop.md` | Fixed A/D axis directions, `lsjs` typo |
| `docs/tutorial/06-instructor-runbook.md` | Fixed COST_LOOKUP values, config filename |
