#!/bin/bash
#SBATCH --job-name=9b-pilot-z2-nogc
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=capella

DIR=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B
export PILOT_STRATEGY=zero2-nogc
export PILOT_CONFIG=${DIR}/config_pilot_zero2_nogc.yaml
export ACCELERATE_CONFIG_FILE=${DIR}/zero2.yaml

bash ${DIR}/pilot_launch.sh
