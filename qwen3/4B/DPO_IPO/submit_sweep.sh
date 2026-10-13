#!/bin/bash
set -euo pipefail
# Submit job scripts in this directory. Skips APO-down. Does not regenerate configs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
fi
shopt -s nullglob
jobs=("${SCRIPT_DIR}"/*.sh)
count=0
for job in "${jobs[@]}"; do
  base="$(basename "$job")"
  if [[ "$base" == "submit_sweep.sh" || "$base" == APOdown-* ]]; then
    continue
  fi
  count=$((count + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN: sbatch $job"
  else
    sbatch "$job"
  fi
done
echo "$count job scripts."
