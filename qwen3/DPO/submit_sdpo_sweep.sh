#!/bin/bash
set -euo pipefail

ROOT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO
SBATCH_SCRIPT="$ROOT/sdpo_LR5e7_Beta1.sh"

CONFIGS=(
  "$ROOT/config_sdpo_LR5e7_Beta1_k1.0_tau0.99_ref2.0_pol1.0_ag0.75_len0.25_eps0.20.yaml"
  "$ROOT/config_sdpo_LR5e7_Beta1_k2.0_tau0.99_ref2.0_pol1.0_ag0.75_len0.25_eps0.20.yaml"
  "$ROOT/config_sdpo_LR5e7_Beta1_k1.0_tau0.995_ref2.0_pol1.0_ag0.75_len0.25_eps0.20.yaml"
  "$ROOT/config_sdpo_LR5e7_Beta1_k1.0_tau0.99_ref1.75_pol1.0_ag0.75_len0.25_eps0.20.yaml"
  "$ROOT/config_sdpo_LR5e7_Beta1_k1.0_tau0.99_ref2.0_pol1.0_ag0.75_len0.25_eps0.15.yaml"
)

for config in "${CONFIGS[@]}"; do
  tag=$(basename "$config" | sed -E 's/config_(.*)\.yaml/\1/')
  echo "Submitting SDPO job for $tag"
  sbatch --export=ALL,CONFIG_FILE="$config" --job-name="$tag" "$SBATCH_SCRIPT"
done
