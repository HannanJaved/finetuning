#!/bin/bash
# 100% English, 0% other languages
# 1 epoch, 2 nodes, 8 H100s. Account cns144, QoS acc_resa, partition acc.
# Submit: sbatch dpo-en100.sh
#SBATCH --job-name=dpo-en100
#SBATCH --account=cns144
#SBATCH --qos=acc_resa
#SBATCH --partition=acc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/logs/%x-%j.out
#SBATCH --error=/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/logs/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
exec bash "/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/launch_dpo.sh" "/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/dpo-en100.yaml"
