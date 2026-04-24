#!/bin/bash
#SBATCH --job-name=Qwen3-Entropy
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/.logs/Qwen3/entropy/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/.logs/Qwen3/entropy/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=alpha

set -euo pipefail

module load release/24.10
module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-openjury/bin/activate

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Fix vLLM V1 engine multi-processing crashes on SLURM
export VLLM_USE_V1=0
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export HF_HOME="/data/horse/ws/hama901h-BFTranslation/.cache"
export HF_DATASETS_CACHE="/data/horse/ws/hama901h-BFTranslation/.cache"

RUN_DIR=${RUN_DIR:?"Set RUN_DIR to the SFT run directory (Qwen3-*-SFT-LR*)"}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-4}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.9}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}
RUN_NAME=${RUN_NAME:-$(basename "$RUN_DIR")}
RESULT_BASE=${RESULT_BASE:-"/data/horse/ws/hama901h-BFTranslation/finetuning/qwen3/entropy/results"}
RESULT_DIR=${RESULT_DIR:-"${RESULT_BASE}/${MODEL_SIZE:-unknown}"}
OUTPUT_CSV=${OUTPUT_CSV:-"${RESULT_DIR}/${RUN_NAME}.csv"}
PROBE_CACHE=${PROBE_CACHE:-"/data/horse/ws/hama901h-BFTranslation/finetuning/qwen3/entropy/probe_prompts.jsonl"}

ROOT_DIR=$(dirname "$RUN_DIR")
RUN_GLOB=$(basename "$RUN_DIR")

mkdir -p "$RESULT_DIR"

python /data/horse/ws/hama901h-BFTranslation/finetuning/qwen3/entropy/compute_entropy.py \
  --checkpoint-root "$ROOT_DIR" \
  --checkpoint-glob "$RUN_GLOB" \
  --checkpoint-subglob "checkpoint-*" \
  --latest-only \
  --dataset "nvidia/Nemotron-Post-Training-Dataset-v2" \
  --dataset-config "default" \
  --dataset-split "chat" \
  --prompt-fraction 0.01 \
  --use-vllm \
  --vllm-tensor-parallel-size "$VLLM_TP_SIZE" \
  --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
  --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --probe-cache "$PROBE_CACHE" \
  --output-csv "$OUTPUT_CSV"
