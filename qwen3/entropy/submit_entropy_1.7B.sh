#!/bin/bash
set -euo pipefail

SIZE="1.7B"
ROOT="/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-${SIZE}-Base/SFT"
JOB_SCRIPT="/data/horse/ws/hama901h-BFTranslation/finetuning/qwen3/entropy/run_entropy_job.sh"
LRS=(1e5 1e6 3e5 5e4 5e5)

for lr in "${LRS[@]}"; do
  run_dir="$ROOT/LR${lr}"
  [ -d "$run_dir" ] || continue
  run_name=$(basename "$run_dir")
  job_name="entropy-${SIZE}-LR${lr}"
  result_dir="/data/horse/ws/hama901h-BFTranslation/finetuning/qwen3/entropy/results/${SIZE}"
  sbatch --job-name "$job_name" \
    --export=ALL,RUN_DIR="$run_dir",RUN_NAME="$run_name",MODEL_SIZE="$SIZE",RESULT_DIR="$result_dir",VLLM_TP_SIZE=4 \
    "$JOB_SCRIPT"
done
