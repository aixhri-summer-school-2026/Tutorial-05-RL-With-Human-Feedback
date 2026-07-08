# Part 1 — MILE on MetaWorld (peg-insert-side)

**Goal:** watch MILE train on the classic peg-insertion benchmark using the loss *you*
implemented, and see the policy's success rate climb round over round. This is the
"MILE works" guarantee — a clean simulator, an automated expert, no human in the loop.

By now you've done [00-setup.md](00-setup.md), read [01-concepts.md](01-concepts.md),
and worked through the three exercises
([02a](02a-scripted-intervener.md) → [02b](02b-loss-exercise.md) → [02c](02c-intervention-model.md)).

**All commands run inside the container:** `make up && make shell`.

---

## Land & Verify

Before anything else, confirm your setup:

```bash
make tutorial-check
```

Expected:
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

Any `FAIL` or `MISSING` → stop and call an instructor.

> **Tip:** start `make sim-up` in a second host terminal now. The Franka sim
> ([05-franka-sim.md](05-franka-sim.md)) takes ~20 s to boot; kicking it off while
> MetaWorld trains means it's ready when you get there.

---

## The environment: `peg-insert-side-v2`

MetaWorld is a benchmark of 50 robot-manipulation tasks built on a simulated **Sawyer**
arm in MuJoCo. We use one task, **peg-insert-side**: the arm must grasp a peg and insert
it into a hole on the side of a box. It's a good MILE testbed because it's *hard enough
to fail often* (precise insertion) but *fast to simulate*.

| Property | Value |
|---|---|
| Robot | Sawyer, 7-DoF, end-effector pointing down |
| **Action** | 4-D continuous: `[Δx, Δy, Δz, gripper]`, each in `[-1, 1]` |
| Raw observation | 39-D MetaWorld state (EE pose, gripper, object & goal poses) |
| **Wrapped observation** | `FrameStack(4)` → `FlattenObservation` = **156-D** (4 frames of temporal context) |
| Reward | dense shaped reward (used only for logging here, *not* for MILE training) |
| Success signal | `info['success'] == 1` when the peg is seated |
| Episode length | truncated at **500** steps |

The wrapping (`FrameStack(4)+Flatten`) is done in `build_franka_or_metaworld_env()`
(`scripts/train_mile.py`). MILE never sees the reward — it learns purely from the
intervention flag ν and the expert's actions.

---

## What Part 1 actually does

This is a **fully synthetic** MILE loop. There is no human. A scripted "human" is
assembled from three pretrained networks and MILE's own intervention model, and *it*
decides when to take over. The point is to isolate and validate **your loss** on a clean
signal: if the loss is correct, success must go up.

Each **round** does three things (`iterative_training()` in `scripts/train_mile.py`):

1. **Collect** — run the current policy in the env for `episodes_per_round` episodes. At
   every step, the synthetic human decides whether to intervene (via
   `computational_intervention_model`); if it does, it substitutes the expert's action
   and records ν=1, otherwise it records the policy's own action with ν=0.
2. **Train** — accumulate the new data with all prior rounds and train for `num_epochs`
   epochs with **your** `mile_cont_loss_fn` (BCE on ν + Gaussian NLL of the expert action
   on ν=1 steps).
3. **Evaluate** — roll out the trained policy for `rollout.n_episodes` episodes (no
   intervention) and report `success_rate`. This is the number that should climb.

---

## The four pretrained models (and how the paper trained them)

The synthetic human needs models that Part 1 loads from `trained_models/`. These are the
paper's own artifacts (downloaded once by the instructor via `gdown`, then committed
in-tree):

| Model | Type | Role in the loop | How it was trained (original paper) |
|---|---|---|---|
| `initial_policy` | SAC | the **policy π_θ** we start from and improve | RL (SAC) stopped early → a *mediocre* agent that fails often |
| `expert_policy` | SAC | the **oracle intervener** — supplies the "human" action on ν=1 steps | RL (SAC) trained to convergence → near-perfect peg-insert |
| `gt_mental_model` | BC (ActorCriticPolicy) | ground-truth mental model π̃ used *during collection* to decide when the synthetic human intervenes | behavior-cloned to model what the human expects the robot to do |
| `warm_started_mental_model` | BC (ActorCriticPolicy) | initial weights for the mental model we *train* | BC warm-start so the mental model doesn't start from scratch |

At **deployment** only `initial_policy` (now improved into π_θ) is kept. The expert and
both mental models exist only to manufacture training data — they stand in for the human
you'll *be* in [05-franka-sim.md](05-franka-sim.md).

`computational_intervention_model` turns "policy vs. mental-model disagreement" into
`p(ν=1|s)` using two per-task numbers from `COST_LOOKUP` in `mile/computational_model.py`:

```python
'peg-insert-side-v2': [75, 175.0],   # [cost, cdf_scale]
```

---

## Running it: input → output

```bash
make tutorial-metaworld
```

This runs `scripts/tutorial_train.py --config config/tutorial_metaworld.yaml`, which
monkey-patches your loss / intervention-model / scripted-rule (or their answer keys) into
the trainer, then calls `train_mile.main()`.

**Inputs**
- `config/tutorial_metaworld.yaml` (2 rounds × 50 epochs, batch 1024, lr 1e-4, `n_episodes: 10` eval)
- the four `trained_models/` artifacts above
- your three exercise implementations (or answer keys)

**Outputs**
- per-batch / per-epoch loss tables (`cont_loss`, `disc_loss`, `total_loss`) streaming to stdout
- a `success_rate` line at each round's evaluation
- the best policy saved under `output_dir/` (`on_best_rollout_success_rate: true`)
- per-round raw data: `output_dir/intervention_data_round<N>.pkl`
- eval videos under `output_dir/metaworld_eval_videos/`

The first three lines tell you which implementations are live:
```
[tutorial] your loss is missing/incorrect — applying the answer key so you can train.
[tutorial] intervention model not yet implemented — applying the answer key so you can train.
[tutorial] scripted intervener not implemented — using the reference rule.
```
Seeing these is fine — training continues with the reference solutions.

**Evaluate a saved policy separately** (100 episodes, deterministic):
```bash
make eval-metaworld                       # defaults to MODEL=output_dir EPISODES=100
make eval-metaworld MODEL=output_dir EPISODES=50
```
It prints `Average score ...` and `Success rate: ...`.

### Continue where you left off (resume)

Each round saves the policy, mental model, and accumulated dataset into `output_dir`. You
don't have to restart from round 0 every time — point `RESUME` at that directory to run
`num_rounds` **more** rounds on top of it:

```bash
make tutorial-metaworld RESUME=../output_dir
```

(The path is `../output_dir` because the trainer runs from `scripts/`.) On startup you'll
see, e.g.:
```
[resume] continuing from '../output_dir' at round 6 (1755 prior transitions)
```

It reloads the saved policy + mental model, reloads the latest
`accumulated_dataset_round<N>.pkl`, and starts at round `N+1` so nothing gets clobbered.
This is how you extend a 2-round run to 4, 6, 8 rounds incrementally instead of committing
to a long run up front. (You can also set `experiment.resume_from` in the config directly;
`RESUME` just overrides it via the `MILE_RESUME_FROM` env var.)

---

## Expected results (measured on this repo, RTX 5070)

The default config is **2 rounds**. A representative run:

| Checkpoint | `success_rate` |
|---|---|
| initial policy (before training) | 0.10 |
| after round 0 | 0.20 |
| after round 1 | 0.50 |

**Success climbs — that's your proof the loss is correct.** Because evaluation uses only
10 episodes, the number is *noisy* (±0.2 run to run) and not strictly monotonic; look at
the trend, not any single value.

Running **6 rounds** (see "Tuning" below) shows the fuller curve — a real measured run on
this machine:

| Round | init | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| `success_rate` | 0.20 | 0.20 | 0.20 | 0.60 | 0.70 | 0.70 | **0.80** |

Improvement is not linear: the first rounds barely move, then it breaks through. This is
normal for the small tutorial dataset (1 episode/round) — the policy needs a couple of
rounds of intervention data before the corrections take hold.

### How this differs from the paper's full run

The paper reproduction (`config/metaworld.yaml`, from the original `config.json`) is much
heavier and drives peg-insert to near-mastery:

| | tutorial (`tutorial_metaworld.yaml`) | paper (`metaworld.yaml`) |
|---|---|---|
| rounds | 2 | 20 |
| epochs/round | 50 | 300 |
| episodes/round collected | 1 | 1 |
| eval episodes | 10 | 30 |
| logging | stdout | wandb |
| purpose | validate your loss in ~3 min | reproduce paper numbers |

The tutorial is a deliberately scaled-down slice: same code, same loss, fewer
rounds/epochs so it finishes inside a session.

---

## Runtime & how to speed it up

**The default `make tutorial-metaworld` (2 rounds) finishes in ~2.5 min** in-container on a
modern desktop CPU — startup ~6 s + initial eval ~5 s + 2 rounds × ~70 s. If you see loss
tables scrolling, it's running, not stuck.

> **Note on GPU:** the image ships **CPU-only torch** on purpose — the MLP is tiny and the
> real cost is MuJoCo env stepping + the intervention model's Monte-Carlo sampling, both
> CPU-bound. The GPU (`gpus: all`) accelerates *rendering* (sim-gui), not training, so these
> times don't change with a GPU. A slower CPU scales them roughly proportionally.

**Measured wall-clock (in-container), scaling with rounds and eval size:**

| | per round | 3-round total |
|---|---|---|
| default (`n_episodes: 10`) | ~55–75 s | ~3.5 min |
| `n_episodes: 5` | ~25–65 s | ~2.2 min (≈40% faster) |

The surprise: **training is trivial** — collect + 50 epochs ≈ **2 s per round** (the
dataset is tiny, one batch per epoch). The **`auto_eval` rollout is ~95% of the time**:
10 episodes × up to 500 MuJoCo steps, and *failed* episodes run the full 500 steps, which
is why early rounds (more failures) are slower than late ones.

**Levers, from safest to most aggressive** (`config/tutorial_metaworld.yaml`,
under `experiment.rollout`):

| Change | Effect on time | Effect on the demo |
|---|---|---|
| `n_episodes: 10 → 5` | ~40% faster | success readout noisier (±0.3), trend still visible |
| eval only the final round | near-linear speedup | you only see before/after, not the curve |
| `auto_eval: false` | rounds drop to ~2 s each | **no `success_rate` at all** — defeats the point; only use if you'll eval once at the end with `make eval-metaworld` |

Reducing `num_epochs` barely helps (training is already ~2 s). On CPU (no GPU) every
number above is roughly 3–5× larger, so `n_episodes: 5` is worth it there.

---

## What you can tune (and what to expect)

All in `config/tutorial_metaworld.yaml` unless noted.

| Knob | Where | Higher → | Expected effect |
|---|---|---|---|
| `num_rounds` | `experiment` | more collect→train cycles | more improvement, roughly diminishing after ~4–6 rounds |
| `episodes_per_round` | `experiment` | more data per round | steadier climb, slower rounds |
| `num_epochs` | `train` | more gradient steps per round | better fit; too high can overfit the tiny dataset |
| `batch_size` | `train` | — | dataset is small; 1024 = ~1 batch/epoch already |
| `lr` | `train` | faster steps | too high destabilizes the NLL term |
| `lambda1` / `lambda2` | `train` | weight action-NLL / intervention-BCE | see the ablation table in [05-franka-sim.md](05-franka-sim.md#extension-ablate-the-loss-weights) |
| `rollout.n_episodes` | `experiment` | more eval episodes | less noisy `success_rate`, slower |
| `cost` | `COST_LOOKUP` in `mile/computational_model.py` | human intervenes **less** (policy must diverge more before p(ν=1) rises) | fewer ν=1 labels → weaker action signal; too high ≈ no supervision |
| `cdf_scale` | `COST_LOOKUP` | sharper CDF transition | how abruptly p(ν=1) switches near the threshold |

### Try it: tune the intervention cost

Open `mile/computational_model.py` and change
`'peg-insert-side-v2': [75, 175.0]` to `[150, 175.0]` (double the cost). Re-run
`make tutorial-metaworld`. The synthetic human now intervenes less, so each round yields
fewer ν=1 corrections — does success still climb? Does it climb slower?

**Questions to consider:**
- How would you pick `cost` for a brand-new task you've never run?
- If `p(ν=1|s)` is near 0 everywhere, what happens to the action-NLL term?
- Is there a `cost` that makes MILE equivalent to plain BC on the interventions?

---

## The takeaway

> "Success went up because the synthetic human's corrections trained the policy through
> **the loss I wrote**." In [05-franka-sim.md](05-franka-sim.md) you replace that
> synthetic human with *yourself*.

## Next step

→ **[04-franka-fake.md](04-franka-fake.md)** — smoke-test the Franka environment (no physics, no ROS).
