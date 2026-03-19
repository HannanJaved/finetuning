#!/bin/bash
set -euo pipefail

ROOT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO
SBATCH_SCRIPT="$ROOT/emdpov2_LR5e7_Beta1.sh"

# Sweep over:
#   policy_gate : {disabled, moderate, stronger} — policy warmup/ramp choices (steps or ratios)
#   eps         : {0.10, 0.20}                    — noise prior for the reliability posterior
#   ag_scale    : {0.75, 1.50}                    — agreement coefficient (larger now that signal is soft+normalized)
#
# The ema0.00-labeled run isolates batch-norm + soft-agreement from any policy gating.
# The ag1.50 run tests whether the better-calibrated soft agreement can carry more weight.
CONFIGS=(
  "$ROOT/config_emdpov2_LR5e7_Beta1_ref2.0_pol1.0_ag0.75_len0.25_eps0.20_ema0.10.yaml"
  "$ROOT/config_emdpov2_LR5e7_Beta1_ref2.0_pol1.0_ag0.75_len0.25_eps0.20_ema0.00.yaml"
  "$ROOT/config_emdpov2_LR5e7_Beta1_ref2.0_pol1.0_ag0.75_len0.25_eps0.20_ema0.30.yaml"
  "$ROOT/config_emdpov2_LR5e7_Beta1_ref2.0_pol1.0_ag0.75_len0.25_eps0.10_ema0.10.yaml"
  "$ROOT/config_emdpov2_LR5e7_Beta1_ref2.0_pol1.0_ag1.50_len0.25_eps0.20_ema0.10.yaml"
)

for config in "${CONFIGS[@]}"; do
  tag=$(basename "$config" | sed -E 's/config_(.*)\.yaml/\1/')
  echo "Submitting EM-DPO v2 job for $tag"
  sbatch --export=ALL,CONFIG_FILE="$config" --job-name="$tag" "$SBATCH_SCRIPT"
done
