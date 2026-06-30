#!/bin/bash
#SBATCH --job-name=9b-pilot-fsdp2
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=capella

export PILOT_STRATEGY=fsdp2
export ACCELERATE_CONFIG_FILE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B/fsdp2_2node.yaml

bash /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B/pilot_launch.sh
