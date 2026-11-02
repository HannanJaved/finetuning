#!/bin/bash
set -euo pipefail
# Submit every job script in this directory.
# Usage: ./submit_sweep.sh [account]
# Example: ./submit_sweep.sh p_neurasearch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${1:-}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
  ACCOUNT=""
elif [[ "${2:-}" == "--dry-run" || "${2:-}" == "-n" ]]; then
  DRY_RUN=1
fi
shopt -s nullglob
count=0
for job in "${SCRIPT_DIR}"/*.sh; do
  base="$(basename "$job")"
  if [[ "$base" == "submit_sweep.sh" ]]; then
    continue
  fi
  count=$((count + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ -n "$ACCOUNT" ]]; then
      echo "DRY RUN: sbatch --account=$ACCOUNT $job"
    else
      echo "DRY RUN: sbatch $job"
    fi
  elif [[ -n "$ACCOUNT" ]]; then
    sbatch --account="$ACCOUNT" "$job"
  else
    sbatch "$job"
  fi
done
echo "$count job scripts."
