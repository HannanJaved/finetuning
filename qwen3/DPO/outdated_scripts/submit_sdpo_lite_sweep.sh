#!/bin/bash
set -euo pipefail

SCRIPT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/sdpo_lite_LR5e7_Beta1.sh
CONFIG_DIR=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO

configs=(
  config_sdpo_lite_LR5e7_Beta1_ret0.2_tau0.99.yaml
)

for cfg in "${configs[@]}"; do
  echo "Submitting $cfg"
  sbatch --export=ALL,CONFIG_FILE=${CONFIG_DIR}/${cfg} "$SCRIPT"
done
