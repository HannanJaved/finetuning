#!/bin/bash
set -euo pipefail

SCRIPT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/sdpo_lite_LR5e7_Beta1.sh
CONFIG=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/config_sdpo_lite_LR5e7_Beta1_ret0.2to0.05_decay_tau0.99.yaml

echo "Submitting SDPO-Lite decay run with config $(basename "$CONFIG")"
sbatch --export=ALL,CONFIG_FILE=${CONFIG} "$SCRIPT"
