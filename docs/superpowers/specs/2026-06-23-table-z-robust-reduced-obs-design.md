# Table-z-robust reduced-observation base policy — design

Date: 2026-06-23
Status: implemented 2026-06-23 (pending recollect + retrain on hardware)
Scope: the Franka block-stacking base policy obs/training/inference pipeline.
Does **not** touch the upstream MILE-on-MetaWorld path.

## Problem

Two coupled problems surfaced while evaluating the BC base policy:

1. **Eval deadlock (already mitigated).** A deterministic closed-loop rollout freezes
   when the policy's mean action collapses to ~0 at out-of-distribution / grasp states:
   `FrankaEnv.step` re-anchors the target to the current EE, so a ~0 delta produces no
   motion → identical obs → the same ~0 action forever. Root cause: the mediocre demos are
   multimodal/aliased at the grasp (descend / hold / lift labels at near-identical poses),
   so BC fits a near-zero conditional mean there. See
   [`memory/base-policy-stall-root-cause.md`].

2. **Table-height / perception fragility on real hardware.** The policy was trained with a
   fixed table height and constant cube orientation. On the real FR3 the table may sit at a
   different z in the base frame, and AprilTag reports measured (noisy, offset) cube
   orientation and z. The previous mitigation (`MaskedNormalizeFeaturesExtractor` zeroing
   the quats + `bottom_z`, plus a fixed-offset `CanonicalCubePoseSource`) assumed the table
   height is known exactly. See [`memory/eval-real-obs-mismatch.md`].

## Goals

- Make the base policy robust to (a) unknown table height, (b) per-cube z perception noise,
  (c) variable transit/lift altitude, and (d) small placement-height variation.
- Shrink the observation to only the dimensions that carry task signal.
- Keep the MetaWorld path untouched.

## Already applied (symptom fixes, committed separately from this design)

- Stochastic rollout sampling (`deterministic=False`) at the four Franka rollout sites:
  `scripts/build_base_policy.py`, `scripts/eval_base_policy_sim.py`,
  `scripts/eval_base_policy_real.py`, `mile_franka/collect.py`. Breaks the deadlock.
  `mile/algorithm.py` (upstream success evaluator, guarded off on Franka) left deterministic.
- `ScriptedPolicyConfig.action_noise_std` 0.0 → 0.02 (small demo jitter; the longer
  horizon below is the real fix for the grasp aliasing).

## Design

### 1. Reduced, configurable observation

`FrankaEnv` emits a reduced frame instead of the 18-dim
`[ee_xyz, grip, top_pose(7), bottom_pose(7)]`:

- **Default 9-dim frame:** `[ee_xyz(3), grip(1), top_xyz(3), bottom_xy(2)]`.
  Dropped: both cube quaternions (cubes don't rotate) and `bottom_z` (constant — the bottom
  cube is always on the table, so its corrected z is the fixed canonical plane).
- `FrameStack(N)` with **N = 10** by default → **90-dim** policy input. The longer temporal
  window (1.0 s at 10 Hz vs 0.4 s) disambiguates descend / hold / lift at the grasp, which
  is the root-cause fix for problem (1).
- Frame composition and stack depth are configuration: `--frame_stack` (default 10) and
  `--include_bottom_z` (default false) in `build_base_policy`, threaded through the env /
  registration so collection, training, and eval agree on the shape.

### 2. Feature extractor

With no constant dims left in the obs, `MaskedNormalizeFeaturesExtractor` is unnecessary on
the Franka path. Use plain `NormalizeFeaturesExtractor` + `RunningNorm`. `features.py` and
`tests/test_masked_features.py` retire from the Franka path (kept only if still referenced
by the MetaWorld path; otherwise removed). `train_mile.py`'s extractor selector returns the
plain extractor for Franka envs.

### 3. Inference z-correction (real backend only)

The table plane is observed directly: the bottom cube always rests on it. Define
`train_rest_z = cube_size / 2` (the resting cube-center z the policy trained on, with the
sim's `table_z = 0`).

- **Table estimate.** `table_est` = the bottom cube's measured center z, denoised by
  averaging in the top cube **only while it is resting** (its z is within `snap_tol` of the
  bottom cube). Once the top cube lifts, it is excluded from the estimate.
- **Offset.** `δ = train_rest_z − table_est`.
- **Apply continuously.** `ee_z` and `top_z` in the observation are shifted by `+δ`
  (`top_z_corrected = raw_top_z + δ`). The top/lifted cube is **never snapped** — it is
  tracked continuously so the lift is visible from the first millimeter. At rest the top
  reads ≈ `train_rest_z`; lifted by `h` it reads `train_rest_z + h`.
- **No obs-snapping in the default frame.** `bottom_z` is not in the 9-dim obs, so there is
  nothing to pin. Snapping `bottom_z` to `train_rest_z` applies **only** in the optional
  10-dim variant (`--include_bottom_z`).
- **Action / workspace.** Policy actions are EE deltas, invariant to the global z shift; the
  env anchors targets to the real EE and clips against the real-frame workspace, so executed
  motion is already correct in the real frame. z-correction is therefore **obs-only** — no
  action/clip correction is applied (the "correct the z of the action" step is a no-op for
  delta actions).

Component boundaries: `CanonicalCubePoseSource` keeps responsibility for orientation
canonicalization (forcing the training quaternion). z-correction is a real-only concern the
env applies in `_build_obs` (`ee_z`, `top_z`) and `step` (clip frame), driven by a small
table-plane estimator that consumes the bottom (and optionally resting top) cube. For the
sim/fake backends the estimator is absent (identity), preserving the backend-agnostic env.

### 4. Collection-time domain randomization (scripted demos)

Randomized per episode by the scripted state machine so the recorded actions stay
self-consistent (they re-plan to the randomized targets):

- **Transit/lift height:** `hover_height` ~ U[0.10, 0.25] m — "carry high or low" across the
  horizontal segment.
- **Placement height:** release the top cube `+`U[0.002, 0.015] m above the bottom cube's
  top surface (small imperfect-placement gap).
- `action_noise_std = 0.02` (applied) and the existing per-episode xy aim offset retained.

**No table-height augmentation and no separate z-augmentation**: snapping/`δ`-correction maps
the real scene into the training frame exactly (the table is measured directly), and
`ee_z` / `top_z` already have healthy natural variance from motion and lifting, so RunningNorm
is not hypersensitive.

### 5. Consequence

The observation shape changes (72 → 90 default), so existing `sim_demos_*.npz` and the saved
`base_policy` are invalid. The pipeline must be regenerated: `make collect-mediocre` (new
shape + randomization) then `make base-policy`.

## Testing

- Obs-shape + composition-config tests: 9-dim default, 10-dim with `--include_bottom_z`,
  `frame_stack` honored end-to-end (env → FrameStack → policy input).
- z-correction tests (`tests/test_canonical_pose.py` + `tests/test_reduced_obs.py`): `δ` from a
  denoised table estimate; `ee_z`/`top_z` shift continuously and monotonically through the
  snap_tol boundary; the lifted top cube is tracked (not snapped); lifted cube excluded from the
  table estimate; the env applies the canonicalizer only to the vertical channels.
- Scripted-policy randomization tests: `hover_height` and placement-offset ranges honored;
  demos remain successful under `require_success` collection.
- `make pose-test` stays green.

## Out of scope

- Upstream MILE-on-MetaWorld obs/extractor/horizon.
- FoundationPose / non-AprilTag perception seams (the `δ`-correction consumes whatever pose
  source is configured).
- Retuning `COST_LOOKUP` for the new obs (separate calibration task).
