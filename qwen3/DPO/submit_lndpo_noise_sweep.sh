#!/bin/bash
set -euo pipefail

ROOT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO
SBATCH_SCRIPT="$ROOT/lndpo_LR5e7_Beta1.sh"

CONFIGS=(
  # "$ROOT/config_lndpo_LR5e7_Beta1_eps0.25.yaml"
  # "$ROOT/config_lndpo_LR5e7_Beta1_eps0.35.yaml"
  # "$ROOT/config_lndpo_LR5e7_Beta1_eps0.40.yaml"
  "$ROOT/config_lndpo_LR5e7_Beta1_eps0.45.yaml"
  "$ROOT/config_lndpo_LR5e7_Beta1_eps0.50.yaml"
)

for config in "${CONFIGS[@]}"; do
  eps=$(basename "$config" | sed -E 's/.*_eps([0-9]+\.[0-9]+)\.yaml/\1/')
  echo "Submitting LN-DPO noise sweep job for eps=$eps"
  sbatch --export=ALL,CONFIG_FILE="$config" --job-name="lndpo_eps${eps}" "$SBATCH_SCRIPT"
done