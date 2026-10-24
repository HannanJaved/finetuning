#!/bin/bash
# Submit all six DPO language-mix jobs.
#
# Before the first submission, on a login node:
#   module load git-lfs/3.4.1
#   cd /gpfs/projects/ehpc533/users/hannan/data/Dolci-Instruct-DPO-translated
#   git lfs pull
#
#   export PATH=/gpfs/projects/ehpc533/users/hannan/.local/bin:$PATH
#   cd /gpfs/projects/ehpc533/users/hannan/post-training
#   uv sync --extra trl
#
# Then:
#   bash /gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/submit_all.sh
# Or submit one mix:
#   sbatch /gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/dpo-en100.sh
set -euo pipefail
cd "/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude"
mkdir -p logs
for spec in dpo-en100.sh dpo-en90.sh dpo-en80.sh dpo-en70.sh dpo-en60.sh dpo-en50.sh; do
  sbatch "${spec}"
done
