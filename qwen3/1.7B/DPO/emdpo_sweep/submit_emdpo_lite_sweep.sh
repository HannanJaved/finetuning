#!/bin/bash
set -euo pipefail

ROOT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/emdpo_sweep
SBATCH_SCRIPT="$ROOT/emdpo_lite_LR5e7_Beta1.sh"

CONFIGS=(
  "$ROOT/config_emdpo_lite_LR5e7_Beta1_ref0.5.yaml"
  "$ROOT/config_emdpo_lite_LR5e7_Beta1_ref1.0.yaml"
  "$ROOT/config_emdpo_lite_LR5e7_Beta1_ref2.0.yaml"
  "$ROOT/config_emdpo_lite_LR5e7_Beta1_ref3.0.yaml"
  "$ROOT/config_emdpo_lite_LR5e7_Beta1_ref1.0_pol1.0.yaml"
  "$ROOT/config_emdpo_lite_LR5e7_Beta1_ref1.5_pol0.5.yaml"
  "$ROOT/config_emdpo_lite_LR5e7_Beta1_ref2.0_pol1.0.yaml"
  "$ROOT/config_emdpo_lite_LR5e7_Beta1_ref2.0_pol2.0.yaml"
)

for config in "${CONFIGS[@]}"; do
  tag=$(basename "$config" | sed -E 's/config_(.*)\.yaml/\1/')
  echo "Submitting EM-DPO-Lite job for $tag"
  sbatch --export=ALL,CONFIG_FILE="$config" --job-name="$tag" "$SBATCH_SCRIPT"
done
