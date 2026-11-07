#!/bin/bash
# Run on an MN5 login node: submit one GRPO run per starting point.
# Usage: bash submit_train_mn5.sh [--submit] [alias ...]
#        no aliases = all five in starts.tsv; without --submit it only prints the sbatch commands
set -euo pipefail
R=/gpfs/projects/ehpc533/users/hannan/rlvr

SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then SUBMIT=1; shift; fi
ALIASES=("$@")
[[ ${#ALIASES[@]} -gt 0 ]] || mapfile -t ALIASES < <(tail -n +2 "$R/code/starts.tsv" | cut -f2)

for alias in "${ALIASES[@]}"; do
    cmd=(sbatch --job-name="rlvr-$alias" "$R/code/train_grpo_mn5.sbatch" "$alias")
    if [[ $SUBMIT -eq 1 ]]; then "${cmd[@]}"; else echo "${cmd[@]}"; fi
done
