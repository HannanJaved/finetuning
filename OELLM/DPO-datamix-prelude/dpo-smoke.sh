#!/bin/bash
# 30-minute smoke test of the 50/50 mix: same queue and 8 GPUs, 2 optimizer steps.
# Submit: sbatch dpo-smoke.sh
#SBATCH --job-name=dpo-smoke
#SBATCH --account=cns144
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/logs/%x-%j.out
#SBATCH --error=/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/logs/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
exec bash "/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/launch_dpo.sh" "/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/dpo-smoke.yaml"
