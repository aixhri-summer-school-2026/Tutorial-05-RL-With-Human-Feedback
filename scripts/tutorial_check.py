"""Tutorial readiness check (run at home and at 0:05). Asserts imports + artifacts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = [
    "trained_models/initial_policy", "trained_models/expert_policy",
    "trained_models/gt_mental_model", "trained_models/warm_started_mental_model",
    "trained_models/franka/base_policy",
]


def main():
    ok = True
    for mod in ("metaworld", "mile", "mile_franka", "torch", "gymnasium"):
        try:
            __import__(mod)
            print(f"  import {mod}: OK")
        except Exception as e:
            ok = False
            print(f"  import {mod}: FAIL ({e})")
    for rel in REQUIRED:
        p = ROOT / rel
        print(f"  artifact {rel}: {'OK' if p.exists() else 'MISSING'}")
        ok = ok and p.exists()
    if not ok:
        sys.exit("tutorial-check FAILED — trained_models/ should be in the repo; re-clone if missing.")
    print("tutorial-check OK — you're ready.")


if __name__ == "__main__":
    main()
