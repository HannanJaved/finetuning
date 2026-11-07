#!/bin/bash
# Submit one GRPO run per starting point in starts.tsv.
# Usage: bash submit_train.sh            (prints the sbatch commands)
#        bash submit_train.sh --submit
set -euo pipefail
HERE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/rlvr_check
mkdir -p /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/RLVR-GSM8K

tail -n +2 "$HERE/starts.tsv" | cut -f2 | while read -r alias; do
    cmd=(sbatch --job-name="rlvr-$alias" "$HERE/train_grpo.sbatch" "$alias")
    if [[ "${1:-}" == "--submit" ]]; then "${cmd[@]}"; else echo "${cmd[@]}"; fi
done
