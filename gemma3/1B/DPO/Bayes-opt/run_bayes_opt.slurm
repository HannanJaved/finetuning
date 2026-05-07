#!/bin/bash
#SBATCH --job-name=Gemma3_1.7b_dpo_bayesopt
#SBATCH --output=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Gemma3/1.7B/DPO/BayesOpt/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Gemma3/1.7B/DPO/BayesOpt/%x_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --partition=capella

set -euo pipefail

echo "JOB NAME" $SLURM_JOB_NAME

module load release/24.10
module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export PYTHONPATH="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/src:/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook:/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"

cd /data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/gemma3/1B/DPO/Bayes-opt/

python bayes_opt.py

echo "END $SLURM_JOBID: $(date)"
