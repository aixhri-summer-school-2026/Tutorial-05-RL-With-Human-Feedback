#!/usr/bin/env bash
# Run ONCE by the instructor. Pulls the MetaWorld expert from Drive, adds the Franka
# base policy, bundles into tutorial-artifacts.tar, prints + writes the SHA256.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK=$(mktemp -d)
echo "[1/4] downloading MetaWorld expert from Drive (one-time)…"
python3 -m gdown 1bzKGyOmX1ZCmAWnZiq_sAFRxi3AXvm4t -O "$WORK/trained_models.zip"
( cd "$WORK" && unzip -q trained_models.zip )   # -> $WORK/trained_models/{4 models}

echo "[2/4] adding the Franka base policy…"
mkdir -p "$WORK/trained_models/franka"
cp trained_models/franka/base_policy "$WORK/trained_models/franka/base_policy"

echo "[3/4] taring the bundle…"
tar -C "$WORK" -cf tutorial-artifacts.tar trained_models

echo "[4/4] checksum…"
sha256sum tutorial-artifacts.tar | tee tutorial-artifacts.sha256
echo "Done. Upload tutorial-artifacts.tar as a GitHub Release asset, commit tutorial-artifacts.sha256."
rm -rf "$WORK"
