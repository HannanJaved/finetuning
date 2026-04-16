#!/bin/bash
#SBATCH --job-name=qwen3_06b_dpo_bayesopt
#SBATCH --output=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/0.6B/DPO/BayesOpt/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/0.6B/DPO/BayesOpt/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --partition=capella

set -euo pipefail

echo "JOB NAME" $SLURM_JOB_NAME

module load release/24.10
module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export PYTHONPATH="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook:/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"

cd /data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/0.6B/DPO/SFT-Lr3e-5/Bayes-opt

python bayes_opt.py

echo "END $SLURM_JOBID: $(date)"
