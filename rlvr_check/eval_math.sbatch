#!/bin/bash
#SBATCH --job-name=math500
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/LM-eval/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/LM-eval/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=capella
#
# MATH-500 (minerva_math500, 4-shot, no chat template), vLLM backend -- the same setup as the
# grid's GSM8K protocol (submit_gsm8k_from_list.py) so pre- and post-RL scores are comparable.
# Usage: sbatch --job-name=math500_<alias> eval_math.sbatch <checkpoint path>
set -euo pipefail
MODEL_PATH=$1
OUT=/data/horse/ws/hama901h-BFTranslation/evaluation_results/minerva_math500

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-vllm/bin/activate
export HF_HOME=/data/horse/ws/hama901h-BFTranslation/.cache
export HF_DATASETS_CACHE=$HF_HOME
export PYTHONPATH=/data/horse/ws/hama901h-BFTranslation/venv-vllm/lib/python3.11/site-packages
cd /data/horse/ws/hama901h-BFTranslation/lm-evaluation-harness
mkdir -p "$OUT"

python -m lm_eval --model vllm \
    --model_args pretrained="$MODEL_PATH",dtype=bfloat16,gpu_memory_utilization=0.9 \
    --tasks minerva_math500 \
    --num_fewshot 4 \
    --gen_kwargs max_new_tokens=1024 \
    --batch_size auto \
    --output_path "$OUT"

echo "END TIME: $(date)"
