#!/bin/bash
set -euo pipefail
# Submit every alpha job script in this directory, skipping any whose output_dir already exists
# (finished, running or partial), so no run can overwrite another. Does not regenerate configs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
fi
shopt -s nullglob
count=0
for job in "${SCRIPT_DIR}"/DPO-*-alpha.sh; do
  cfg="$(grep -m1 '^CONFIG_FILE=' "$job" | cut -d= -f2)"
  out="$(grep -m1 '^output_dir:' "$cfg" | awk '{print $2}')"
  if [[ -d "$out" ]]; then
    echo "SKIP (output_dir exists): $(basename "$job")"
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
