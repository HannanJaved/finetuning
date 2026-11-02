#!/bin/bash
set -euo pipefail
# Submit the high-beta IPO and NormDPO sweeps for 1.7B, 4B, and 8B.
# Usage: ./submit_highbeta_ipo_norm.sh [account] [--dry-run]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=("$@")
for size in 1.7B 4B 8B; do
  for variant in DPO_IPO_HIBETA DPO_NORM_HIBETA; do
    echo "== $size $variant =="
    "$ROOT/$size/$variant/submit_sweep.sh" "${args[@]}"
  done
done
