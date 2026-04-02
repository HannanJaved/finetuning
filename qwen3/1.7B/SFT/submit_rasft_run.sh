#!/bin/bash
set -euo pipefail

SCRIPT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/SFT/rasft_LR5e5.sh
CONFIG=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/SFT/config_rasft_LR5e5.yaml

echo "Submitting RA-SFT run with config $(basename "$CONFIG")"
sbatch --export=ALL,CONFIG_FILE=${CONFIG} "$SCRIPT"
