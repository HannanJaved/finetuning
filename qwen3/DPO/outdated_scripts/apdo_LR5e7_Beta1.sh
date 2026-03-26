#!/bin/bash
#SBATCH --job-name=apdo-qwen3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/%x_%j.err

set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:-/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/config_apdo_LR5e7_Beta1_beta0.10_alpha0.50_tau0.99_mix0.50.yaml}

source /etc/profile || true
module load CUDA/12.4.0 || true

export WANDB_PROJECT=instruction-tuning
export WANDB_WATCH=false
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data/cat/ws/hama901h-Posttraining/finetuning/alignment-handbook/src:${PYTHONPATH:-}

VENV=/data/horse/ws/hama901h-BFTranslation/venv-TRL
SCRIPT=/data/cat/ws/hama901h-Posttraining/finetuning/alignment-handbook/scripts/apdo.py
DEEPSPEED_CONFIG=/data/cat/ws/hama901h-Posttraining/finetuning/alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml

srun --cpu_bind=none ${VENV}/bin/accelerate launch \
  --config_file "$DEEPSPEED_CONFIG" \
  "$SCRIPT" \
  --config "$CONFIG_FILE"
