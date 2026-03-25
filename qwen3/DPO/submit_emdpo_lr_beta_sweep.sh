#!/bin/bash
set -euo pipefail

ROOT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO
SBATCH_SCRIPT="$ROOT/emdpo_LR5e7_Beta1.sh"

LRS=("5e-6")
BETAS=("2.0")
# LRS=("1e-7" "5e-7" "1e-6" "5e-6")
# BETAS=("0.1" "0.25" "0.5" "0.75" "1.0" "2.0")

for lr in "${LRS[@]}"; do
  lr_tag=$(echo "$lr" | sed 's/-//g')
  for beta in "${BETAS[@]}"; do
    config="$ROOT/config_emdpo_LR${lr_tag}_Beta${beta}.yaml"
    tag=$(basename "$config" .yaml)
    tag=${tag#config_}
    tag=${tag//./p}

    if [[ ! -f "$config" ]]; then
      echo "Skipping missing config: $config"
      continue
    fi

    echo "Submitting EM-DPO job for $tag"
    sbatch \
      --export=ALL,CONFIG_FILE="$config" \
      --job-name="$tag" \
      --output=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/${tag}_%j.out \
      --error=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/${tag}_%j.err \
      "$SBATCH_SCRIPT"
  done
done