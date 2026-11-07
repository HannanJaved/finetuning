#!/bin/bash
#SBATCH --job-name=rlvr-gsm8k
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/RLVR-GSM8K/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/RLVR-GSM8K/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --partition=capella
#
# Usage: sbatch --job-name=rlvr-<alias> train_grpo.sbatch <alias> [extra GRPOConfig overrides]
#        (alias = column 2 of starts.tsv; submit_train.sh submits all of them)
# Smoke test: sbatch --time=00:45:00 train_grpo.sbatch <alias> --max_steps 3 --output_dir /tmp/rlvr-smoke
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
    grpo_gsm8k.py --config grpo_gsm8k.yaml --model_name_or_path "$MODEL" --output_dir "$OUT" "${@:2}"

echo "END $ALIAS $(date)"
