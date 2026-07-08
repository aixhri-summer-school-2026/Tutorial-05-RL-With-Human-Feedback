"""Reference answer for the scripted intervener exercise."""
import numpy as np


def should_intervene(expert_action: np.ndarray, rollout_action: np.ndarray) -> bool:
    """Return True if the human should take over, False otherwise.

    expert_action / rollout_action layout: [dx, dy, dz, gripper_command]
    """
    # Cartesian drift: L2 distance between expert and policy translation deltas.
    drift = np.linalg.norm(expert_action[:3] - rollout_action[:3]) > 0.5

    # Gripper disagreement: expert wants open but policy wants closed (or vice versa).
    grip_disagree = (expert_action[3] > 0) != (rollout_action[3] > 0)

    return bool(drift or grip_disagree)