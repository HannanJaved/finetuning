#!/bin/bash
set -euo pipefail

ROOT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO
SBATCH_SCRIPT="$ROOT/emdpo_LR5e7_Beta1.sh"

CONFIGS=(
  "$ROOT/config_emdpo_LR5e7_Beta1_ref2.0_pol1.0_ag0.75_len0.25_eps0.10.yaml"
  # "$ROOT/config_emdpo_LR5e7_Beta1_ref2.0_pol1.0_ag0.75_len0.25_eps0.20.yaml"
  # "$ROOT/config_emdpo_LR5e7_Beta1_ref2.0_pol1.0_ag0.75_len0.25_eps0.30.yaml"
  # "$ROOT/config_emdpo_LR5e7_Beta1_ref1.5_pol1.5_ag1.0_len0.25_eps0.20.yaml"
  # "$ROOT/config_emdpo_LR5e7_Beta1_ref2.5_pol0.5_ag1.0_len0.25_eps0.20.yaml"
)

for config in "${CONFIGS[@]}"; do
  tag=$(basename "$config" | sed -E 's/config_(.*)\.yaml/\1/')
  echo "Submitting EM-DPO job for $tag"
  sbatch --export=ALL,CONFIG_FILE="$config" --job-name="$tag" "$SBATCH_SCRIPT"
done