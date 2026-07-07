# The Scripted Intervener Decision Rule

## The Problem

When MILE uses a synthetic "human" to collect training data, something has to decide at every timestep: should the human step in right now?

In the computational model exercise, you designed `compute_intervention_prob` — a *probabilistic* model of this decision, learned from data.

Here, you'll design the **hard-coded** version: a binary rule used by the scripted intervener during data collection. The scripted intervener compares what the expert *would do* against what the base policy *is doing*, and decides.

**File to edit:**
```
mile_franka/tutorial/scripted_intervener_exercise.py
```

## Your Task

Implement:
```python
def should_intervene(expert_action: np.ndarray, rollout_action: np.ndarray) -> bool:
    # expert_action:  [dx, dy, dz, gripper_command]  — what the expert wants
    # rollout_action: [dx, dy, dz, gripper_command]  — what the policy is doing
    # return True to take over, False to let the policy run
```

## Design Questions

1. **Which dimensions matter most?** Translation (XYZ) drift directly causes the robot to miss a target. Gripper disagreement (open vs. closed at the wrong moment) kills a grasp or release. Are they equally important?

2. **Threshold selection**: what L2 drift in action space corresponds to a "meaningful" mistake in Cartesian space for the Franka? Actions are in ~[-1, 1] and translate to ~2 cm/step.

3. **Per-axis vs. combined**: should you threshold each axis independently or use combined L2 norm?

## Testing Your Implementation

```bash
make tutorial-check-scripted-intervener
```

The three tests check:
- Large translation drift (L2 > 1.0) → should intervene
- Identical actions → should NOT intervene  
- Gripper disagreement with tiny position drift → should intervene

## The Connection to Your Computational Model

Your `compute_intervention_prob` in `intervention_model_exercise.py` is a **learned**, probabilistic version of this same decision. The scripted intervener is a **hand-coded**, deterministic version.

They answer the same question — "is the robot about to do something wrong?" — but with different information:
- **Scripted intervener**: has access to the expert's action directly. Privileged information.
- **Computational model**: only has the policy and mental model distributions. No privileged access.

At test/deployment time, only the computational model is available (there is no oracle expert). The scripted intervener generates the *training data* that teaches the computational model to recognize these situations from distributions alone.

## How It Connects to Training

When `tutorial_train.py` runs with `intervener: scripted`, your `should_intervene` is injected into `ScriptedIntervener.intervene`. The training data that MILE learns from reflects your intervention rule — the data reflects your implicit theory of "what the robot needs help with."

If you define a bad rule (e.g., intervene every step), the policy gets dense corrections but no signal about when it was already doing fine. If you define too permissive a rule (e.g., never intervene), the policy gets no human signal at all. The right rule is the one that mirrors how a real human would actually supervise the robot.

## Next Step

→ **[02b-loss-exercise.md](02b-loss-exercise.md)**: Exercise 2 — implement the training objective.
