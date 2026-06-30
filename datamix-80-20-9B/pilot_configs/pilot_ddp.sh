#!/bin/bash
#SBATCH --job-name=9b-pilot-ddp
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=capella

export PILOT_STRATEGY=ddp
export ACCELERATE_CONFIG_FILE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B/ddp_1node.yaml

bash /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B/pilot_launch.sh
