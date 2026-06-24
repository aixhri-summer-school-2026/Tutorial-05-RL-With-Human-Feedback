"""Tutorial training wrapper: inject the participant's loss (or the answer key), then
run the normal train_mile main. Usage: python scripts/tutorial_train.py --config <cfg>
"""
import sys

import mile.algorithm as algorithm
from mile_franka.tutorial.loss_loader import resolve_loss_fn


def main():
    loss_fn, _ = resolve_loss_fn()
    algorithm.mile_cont_loss_fn = loss_fn  # monkeypatch BEFORE trainer construction
    import train_mile  # scripts/ is on sys.path when run from scripts/
    train_mile.main()


if __name__ == "__main__":
    main()
