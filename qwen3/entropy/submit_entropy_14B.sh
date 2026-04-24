#!/bin/bash
set -euo pipefail

SIZE="14B"
ROOT="/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-${SIZE}-Base/SFT-sweep"
JOB_SCRIPT="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/entropy/run_entropy_job.sh"
LRS=(1e-5 1e-6 3e-5 5e-4 5e-5)

for lr in "${LRS[@]}"; do
  run_dir="$ROOT/Qwen3-${SIZE}-SFT-LR${lr}"
  [ -d "$run_dir" ] || continue
  run_name=$(basename "$run_dir")
  job_name="entropy-${SIZE}-LR${lr}"
  result_dir="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/entropy/results/${SIZE}"
  sbatch --job-name "$job_name" \
    --export=ALL,RUN_DIR="$run_dir",RUN_NAME="$run_name",MODEL_SIZE="$SIZE",RESULT_DIR="$result_dir" \
    "$JOB_SCRIPT"
done
