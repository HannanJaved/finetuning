#!/bin/bash
set -euo pipefail

ROOT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO
SBATCH_SCRIPT="$ROOT/apdo_LR5e7_Beta1.sh"

CONFIGS=(
  "$ROOT/config_apdo_LR5e7_Beta1_beta0.10_alpha0.50_tau0.99_mix0.50.yaml"
  "$ROOT/config_apdo_LR5e7_Beta1_beta0.20_alpha0.50_tau0.99_mix0.50.yaml"
  "$ROOT/config_apdo_LR5e7_Beta1_beta0.10_alpha0.30_tau0.99_mix0.50.yaml"
  "$ROOT/config_apdo_LR5e7_Beta1_beta0.10_alpha0.50_tau0.995_mix0.50.yaml"
  "$ROOT/config_apdo_LR5e7_Beta1_beta0.10_alpha0.50_tau0.99_mix0.70.yaml"
)

for config in "${CONFIGS[@]}"; do
  tag=$(basename "$config" | sed -E 's/config_(.*)\.yaml/\1/')
  echo "Submitting APDO job for $tag"
  sbatch --export=ALL,CONFIG_FILE="$config" --job-name="$tag" "$SBATCH_SCRIPT"
done
