"""Download + verify the tutorial artifacts bundle from the GitHub Release (NOT Drive).

Homework step. Set ARTIFACTS_URL (defaults to the repo's latest-release asset) and run
`make fetch-artifacts`. Verifies SHA256 against tutorial-artifacts.sha256, then unpacks
trained_models/ into the repo root.
"""
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAR = ROOT / "tutorial-artifacts.tar"
SUMS = ROOT / "tutorial-artifacts.sha256"
DEFAULT_URL = os.environ.get(
    "ARTIFACTS_URL",
    "https://github.com/rayray2002/mile-franka-tutorial/releases/latest/download/tutorial-artifacts.tar",
)


def expected_sha():
    line = SUMS.read_text().split()[0]
    if line.startswith("PENDING"):
        sys.exit("tutorial-artifacts.sha256 not finalized — instructor must run "
                 "scripts/make_tutorial_artifacts.sh and commit the checksum.")
    return line


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not TAR.exists():
        print(f"downloading {DEFAULT_URL}")
        subprocess.check_call(["curl", "-fL", "-o", str(TAR), DEFAULT_URL])
    got, want = sha256(TAR), expected_sha()
    if got != want:
        sys.exit(f"checksum mismatch: got {got}, want {want}")
    print("checksum ok; unpacking…")
    with tarfile.open(TAR) as t:
        t.extractall(ROOT)
    print("artifacts ready under trained_models/")


if __name__ == "__main__":
    main()
