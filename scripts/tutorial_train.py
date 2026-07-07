"""Tutorial training wrapper: inject the participant's loss and intervention model
(or their answer keys), then run the normal train_mile main.
Usage: python scripts/tutorial_train.py --config <cfg>
"""
import sys

import numpy as np

import mile.algorithm as algorithm
from mile_franka.collect import ScriptedIntervener
from mile_franka.tutorial.loss_loader import resolve_loss_fn
from mile_franka.tutorial.intervention_model_loader import resolve_intervention_model
from mile_franka.tutorial.scripted_intervener_loader import resolve_should_intervene_fn


def _make_patched_intervene(should_intervene_fn):
    """Wrap should_intervene(expert_action, rollout_action) into the full
    ScriptedIntervener.intervene(self, obs, rollout_action) signature."""
    def patched_intervene(self, obs, rollout_action):
        frame = (self.env.unwrapped.privileged_frame() if self.env is not None
                 else np.asarray(obs)[-18:])
        expert_action = self.expert.act(frame)
        intervene = bool(should_intervene_fn(expert_action, rollout_action))
        return expert_action, intervene, False, False
    return patched_intervene


def main():
    loss_fn, _ = resolve_loss_fn()
    algorithm.mile_cont_loss_fn = loss_fn  # monkeypatch BEFORE trainer construction

    full_model_fn, _ = resolve_intervention_model()
    # algorithm.py does `from mile.computational_model import computational_intervention_model`,
    # so we patch the name in the algorithm module (not the source module).
    algorithm.computational_intervention_model = full_model_fn

    should_intervene_fn, _ = resolve_should_intervene_fn()
    ScriptedIntervener.intervene = _make_patched_intervene(should_intervene_fn)

    import train_mile  # scripts/ is on sys.path when run from scripts/
    train_mile.main()


if __name__ == "__main__":
    main()
