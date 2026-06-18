# Tuning `COST_LOOKUP` — the MILE intervention cost guide (2026-06-18)

How to set `[cost, cdf_scale]` in `mile/computational_model.py:COST_LOOKUP` for a new env
(here: `Franka-Stack-Sim-v0` / `Franka-Stack-Real-v0`). Read this before re-tuning the
Franka entries on sim or hardware.

## What these two numbers actually do

`COST_LOOKUP[env] = [cost, cdf_scale]`. They are **fixed hyperparameters**, *not* learned by
gradient descent. They parameterise the probit intervention model in
`computational_intervention_model` (continuous / `Box` branch), which predicts
`p(ν=1 | s)` — the probability a human intervenes in state `s`.

For the continuous case the per-state intervention probability is

```
z      = E_{a~π}[ logπ(a|s) ]  −  E_{ã~π̃}[ logπ(ã|s) ]  −  cost
p(ν=1) = Φ( z / cdf_scale )           # Φ = standard-normal CDF
```

where `π` is the **policy** and `π̃` is the **mental model** (what the human believes the
robot will do). Both expectations are Monte-Carlo over 1000 samples
(`computational_model.py:110-126`).

Intuition:
- `logπ(a) − E_π̃[logπ]` is a **disagreement / surprise** term. When the policy does roughly
  what the human expects, it is small; when the policy is about to do something the human
  did not anticipate, it grows. This is the part the trainer *learns* (it shapes `π` and `π̃`).
- **`cost`** is the bar the surprise must clear before a human bothers to intervene.
  It is the price of grabbing the controller.
  - ⬆ **higher cost → `z` smaller → lower predicted intervention rate** (intervening is
    "expensive", so the human tolerates more).
  - ⬇ **lower cost → higher predicted intervention rate** (cheap to intervene, human jumps in).
- **`cdf_scale`** is the **softness / temperature** of the decision (the std of the probit).
  - ⬆ larger → flatter sigmoid → `p` pulled toward 0.5 (fuzzy, indecisive predictions).
  - ⬇ smaller → sharper step at the threshold → near-deterministic 0/1 around `z = 0`.

### Where they are consumed

`InterventionTrainer` reads them once (`algorithm.py:181-182`) and passes them into every
`computational_intervention_model` call. The model's predicted `intervention_prob` (a
`[1−p, p]` vector) is then trained against the human's **actual** binary ν via the discrete
BCE/NLL loss (`mile_disc_loss_fn`, `mile_cont_loss_fn`). The Gaussian NLL on the human
action only fires on ν=1 steps.

**Key consequence:** if `[cost, cdf_scale]` are miscalibrated, the model's predicted ν
distribution is systematically biased (always ~0, always ~1, or stuck near 0.5) and *no*
setting of `π, π̃` can drive the discrete loss down. Calibration is a precondition for the
loss to be learnable, not a cosmetic knob.

## Why the numbers are env-specific (and large)

`z` is built from **summed log-probabilities of the full action vector**. For the 4-DoF
Franka action, summed log-probs run to tens–hundreds in magnitude and scale with action
dimensionality and the policy's std. That is why `cost` lives in the tens–hundreds and
`cdf_scale` in the hundreds, and why a value tuned on MetaWorld (`pick-place-v2 = [250,
200]`) is only a *seed* for Franka, not a transferable constant. Current Franka seed:
`[70, 100]` (lowered from the `pick-place` seed `[250, 200]` to predict more intervention).

## The calibration procedure

The target is simple: **the model's mean predicted intervention rate over a held dataset
should match the human's observed intervention fraction**, with predictions that are
confident where they should be (not all stuck at 0.5).

1. **Get a reference intervention rate.** Run one real human-in-the-loop collection round
   (`make mile` with `intervener: joystick`) and read the per-round log line
   `interventions=… / recorded=…` from `Collector` (`collect.py`). Compute
   `observed_rate = Σ ν / Σ steps`. For mediocre-base + attentive-human stacking, expect
   roughly **0.15–0.40**. Save that dataset (`.npz`) so tuning is repeatable offline.

2. **Sweep `cost` to match the rate.** With `π`/`π̃` fixed (the base policy + an
   untrained or one-epoch mental model is fine for a first pass), evaluate
   `computational_intervention_model` over the dataset states and compute
   `mean(intervention_prob[:,1])`. Adjust:
   - predicted rate **too low** vs observed → **decrease `cost`**.
   - predicted rate **too high** → **increase `cost`**.
   Bisect on `cost` until predicted ≈ observed within a few points.

3. **Set `cdf_scale` for confidence.** Inspect the spread of predicted `p` across states:
   - if almost all `p ≈ 0.5` (model can't tell intervene-states from not) → **decrease
     `cdf_scale`** to sharpen.
   - if `p` saturates at 0/1 and the discrete loss plateaus high / training is unstable →
     **increase `cdf_scale`** to soften.
   A healthy distribution is bimodal-ish: low `p` in clearly-safe states, high `p` near the
   failure/contact moments where the human actually grabbed control.

4. **Re-check after training.** `cost`/`cdf_scale` interact with the learned `π, π̃`. After
   a full run, re-plot predicted vs observed ν and the per-round success curve (gate 5). If
   success isn't improving across rounds and the discrete loss is stuck, suspect cost
   miscalibration before anything else.

### Quick diagnostic (no script yet — worth adding)

There is currently **no** calibration script in the repo. A small offline helper
(`scripts/calibrate_cost.py`) is worth adding: load a collected `.npz` + the base policy +
mental model, sweep `cost ∈ {30,50,70,100,150}` × `cdf_scale ∈ {50,100,200}`, and print for
each (predicted mean rate, predicted-rate stddev, discrete loss vs the recorded ν). Pick the
cell whose predicted rate matches `observed_rate` with the widest spread (most decisive).

## Symptom → knob cheat-sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Model predicts ν≈0 everywhere; human clearly intervened a lot | `cost` too high | lower `cost` |
| Model predicts ν≈1 everywhere | `cost` too low | raise `cost` |
| All predictions hug 0.5, loss won't drop | `cdf_scale` too large | lower `cdf_scale` |
| Predictions saturate 0/1, training unstable | `cdf_scale` too small | raise `cdf_scale` |
| Success rate flat across rounds, discrete loss stuck | rate miscalibrated | redo steps 1–3 |

## Hardware note

`Franka-Stack-Real-v0` should **start from the sim-tuned values** but will need a fresh
calibration pass on hardware: real human intervention rate, action-noise scale, and the
base policy's competence all differ from sim. Re-running steps 1–3 is part of Phase (b)/(c)
day-of bring-up. Keep the sim and real entries as **separate** `COST_LOOKUP` keys so they
can diverge (they are already separate). Related: [[teleop-device-joystick]].
