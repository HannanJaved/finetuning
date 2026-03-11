#!/bin/bash
#SBATCH --job-name=dpo_random_3
#SBATCH --output=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/RandomSearch/dpo_random_%j.out
#SBATCH --error=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/RandomSearch/dpo_random_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Posttraining/.cache"
export PYTHONPATH="/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"

export MASTER_PORT=24437
master_addr=
export MASTER_ADDR=
export LOCAL_RANK=
export RANK=
export WORLD_SIZE=0

ACCELERATE_CONFIG_FILE=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/zero3.yaml
CONFIG_FILE=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/RS/configs/config_dpo_random_3.yaml

srun --wait=60 --kill-on-bad-exit=1 bash -c "accelerate launch --config_file  --num_machines  --num_processes  --main_process_ip  --main_process_port  scripts/dpo.py --config /data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/RS/configs/config_dpo_random_3.yaml"
