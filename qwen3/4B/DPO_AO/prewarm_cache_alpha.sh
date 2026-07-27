#!/bin/bash
#SBATCH --job-name=DPO-AO-cache-prewarm
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/DPO-AO/cache-prewarm/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/DPO-AO/cache-prewarm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=alpha

set -euo pipefail

echo "JOB NAME: $SLURM_JOB_NAME"
echo "START TIME: $(date)"

module load release/24.10
module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-post-training/bin/activate

export HF_HOME="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
source /data/horse/ws/hama901h-Post-training/cache.sh
export PYTHONPATH="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/src:/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook:/data/horse/ws/hama901h-BFTranslation/venv-post-training/lib/python3.11/site-packages"

export WANDB_MODE=disabled

cd /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/
CONFIG_FILE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/4B/DPO_AO/dpo_smoketest_alpha.yaml

echo "CONFIG: $CONFIG_FILE"

# Single process, no accelerate/deepspeed/srun: this is the only writer of the
# HF datasets cache, so the real multi-node smoketest only ever reads it back
# instead of racing multiple nodes to build it concurrently.
python scripts/dpo.py --config $CONFIG_FILE

echo "END TIME: $(date)"
