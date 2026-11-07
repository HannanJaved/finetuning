#!/bin/bash
#SBATCH --job-name=rlvr-gsm8k
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/RLVR-GSM8K-alpha/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/RLVR-GSM8K-alpha/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --partition=alpha
#SBATCH --exclusive
#
# Alpha (8x A100) version of train_grpo.sbatch. Same RL run as on capella: the global batch stays
# 8 GPUs x 2 per device x 16 accumulation = 256 completions = 32 prompts x 8 per optimizer step,
# one generation round per step. Only memory knobs change: a quarter of the per-device batch (4 OOMed
# in the first backward on 40 GB: fp32 logits of 4x1024 tokens x 152k vocab) and a larger
# vLLM share, so it also fits 40 GB cards.
# Writes to the same output dir as the capella runs; grpo_gsm8k.py lets only the first job that
# starts train, any later duplicate exits.
#
# Usage: sbatch [--account=...] --job-name=rlvr-<alias> train_grpo_alpha.sbatch <alias> [overrides]
#        (submit_train_alpha.sh submits all five)
# Smoke test: sbatch --time=00:45:00 train_grpo_alpha.sbatch <alias> --max_steps 3 --output_dir <scratch dir>
set -euo pipefail

ALIAS=$1
HERE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/rlvr_check
MODEL=$(awk -F'\t' -v a="$ALIAS" '$2 == a {print $3}' "$HERE/starts.tsv")
OUT=/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-4B-Base/RLVR-GSM8K/$ALIAS
[[ -n "$MODEL" ]] || { echo "alias $ALIAS not in starts.tsv" >&2; exit 1; }
echo "START $ALIAS  $MODEL -> $OUT  $(date)"

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-vllm/bin/activate
source /data/horse/ws/hama901h-Post-training/cache.sh
export PYTHONPATH=/data/horse/ws/hama901h-BFTranslation/venv-vllm/lib/python3.11/site-packages
# openai/gsm8k is cached here (cache.sh points HF_HOME at the Post-training cache, which lacks it)
export HF_HOME=/data/horse/ws/hama901h-BFTranslation/.cache
export HF_DATASETS_CACHE=$HF_HOME
export HF_DATASETS_OFFLINE=1

cd "$HERE"
accelerate launch --config_file accelerate_fsdp.yaml --num_processes "$SLURM_GPUS_ON_NODE" \
    grpo_gsm8k.py --config grpo_gsm8k.yaml \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --vllm_gpu_memory_utilization 0.35 \
    --model_name_or_path "$MODEL" --output_dir "$OUT" "${@:2}"

echo "END $ALIAS $(date)"
