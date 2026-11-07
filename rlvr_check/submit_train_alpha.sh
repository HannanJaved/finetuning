#!/bin/bash
# Run on an alpha login node: submit one GRPO run per starting point in starts.tsv.
# Usage: bash submit_train_alpha.sh [--submit] [extra sbatch options, e.g. --account=p_neurasearch]
#        (without --submit it only prints the sbatch commands)
set -euo pipefail
HERE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/rlvr_check
mkdir -p /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/RLVR-GSM8K-alpha

SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then SUBMIT=1; shift; fi

tail -n +2 "$HERE/starts.tsv" | cut -f2 | while read -r alias; do
    cmd=(sbatch "$@" --job-name="rlvr-$alias" "$HERE/train_grpo_alpha.sbatch" "$alias")
    if [[ $SUBMIT -eq 1 ]]; then "${cmd[@]}"; else echo "${cmd[@]}"; fi
done
