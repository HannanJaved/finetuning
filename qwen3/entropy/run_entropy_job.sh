#!/bin/bash
#SBATCH --job-name=Qwen3-Entropy
#SBATCH --output=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/entropy/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/entropy/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=capella

set -euo pipefail

module load release/24.10
module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"

RUN_DIR=${RUN_DIR:?"Set RUN_DIR to the SFT run directory (Qwen3-*-SFT-LR*)"}
RUN_NAME=${RUN_NAME:-$(basename "$RUN_DIR")}
RESULT_BASE=${RESULT_BASE:-"/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/entropy/results"}
RESULT_DIR=${RESULT_DIR:-"${RESULT_BASE}/${MODEL_SIZE:-unknown}"}
OUTPUT_CSV=${OUTPUT_CSV:-"${RESULT_DIR}/${RUN_NAME}.csv"}
PROBE_CACHE=${PROBE_CACHE:-"/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/entropy/probe_prompts.jsonl"}

ROOT_DIR=$(dirname "$RUN_DIR")
RUN_GLOB=$(basename "$RUN_DIR")

mkdir -p "$RESULT_DIR"

python /data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/entropy/compute_entropy.py \
  --checkpoint-root "$ROOT_DIR" \
  --checkpoint-glob "$RUN_GLOB" \
  --checkpoint-subglob "checkpoint-*" \
  --latest-only \
  --dataset "nvidia/Nemotron-Post-Training-Dataset-v2" \
  --dataset-config "default" \
  --dataset-split "chat" \
  --prompt-fraction 0.01 \
  --probe-cache "$PROBE_CACHE" \
  --output-csv "$OUTPUT_CSV"
