#!/bin/bash
set -euo pipefail

# Alpha DPO follow-up. Each job is 2 nodes, 8 GPUs, grad accumulation 8
# (global batch 128, same as the capella 4-GPU runs).
#
# Default submits the full set, including the optional runs:
#   4B full Dolci, beta 0.04 at 1e-7
#   half Dolci at 2e-7 and 3.2e-6, both sizes
# --core-only keeps the minimum:
#   4B full Dolci, beta 0.01 at 1e-7
#   half Dolci at 4e-7, 8e-7 and 1.6e-6, both sizes, beta 0.01
#
# Half-Dolci jobs share dataset_mixture seed 8 and warmup_ratio 0.1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
CORE_ONLY=0

show_help() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run|-n] [--core-only]

  --dry-run, -n   Print sbatch commands without submitting
  --core-only     Skip the optional beta 0.04 run and the 2e-7 / 3.2e-6 half-Dolci runs
  --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=1
      shift
      ;;
    --core-only)
      CORE_ONLY=1
      shift
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

mkdir -p \
  /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/DPO-AO-DDP/SFT-LR3e-5 \
  /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/8B/DPO-AO/SFT-LR3e-5

jobs=(
  "${SCRIPT_DIR}/4B/DPO-LR1e-7_Beta0.01-Qwen3-4B-SFT-LR3e-5-AO.sh"
  "${SCRIPT_DIR}/4B/half_dolci/DPO-HalfDolci-LR4e-7_Beta0.01-Qwen3-4B-SFT-LR3e-5-AO.sh"
  "${SCRIPT_DIR}/4B/half_dolci/DPO-HalfDolci-LR8e-7_Beta0.01-Qwen3-4B-SFT-LR3e-5-AO.sh"
  "${SCRIPT_DIR}/4B/half_dolci/DPO-HalfDolci-LR1.6e-6_Beta0.01-Qwen3-4B-SFT-LR3e-5-AO.sh"
  "${SCRIPT_DIR}/8B/half_dolci/DPO-HalfDolci-LR4e-7_Beta0.01-Qwen3-8B-SFT-LR3e-5-AO.sh"
  "${SCRIPT_DIR}/8B/half_dolci/DPO-HalfDolci-LR8e-7_Beta0.01-Qwen3-8B-SFT-LR3e-5-AO.sh"
  "${SCRIPT_DIR}/8B/half_dolci/DPO-HalfDolci-LR1.6e-6_Beta0.01-Qwen3-8B-SFT-LR3e-5-AO.sh"
)

if [[ "${CORE_ONLY}" -eq 0 ]]; then
  jobs+=(
    "${SCRIPT_DIR}/4B/DPO-LR1e-7_Beta0.04-Qwen3-4B-SFT-LR3e-5-AO.sh"
    "${SCRIPT_DIR}/4B/half_dolci/DPO-HalfDolci-LR2e-7_Beta0.01-Qwen3-4B-SFT-LR3e-5-AO.sh"
    "${SCRIPT_DIR}/4B/half_dolci/DPO-HalfDolci-LR3.2e-6_Beta0.01-Qwen3-4B-SFT-LR3e-5-AO.sh"
    "${SCRIPT_DIR}/8B/half_dolci/DPO-HalfDolci-LR2e-7_Beta0.01-Qwen3-8B-SFT-LR3e-5-AO.sh"
    "${SCRIPT_DIR}/8B/half_dolci/DPO-HalfDolci-LR3.2e-6_Beta0.01-Qwen3-8B-SFT-LR3e-5-AO.sh"
  )
fi

for job in "${jobs[@]}"; do
  if [[ ! -f "${job}" ]]; then
    echo "Missing job script: ${job}" >&2
    exit 1
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "sbatch ${job}"
  else
    sbatch "${job}"
  fi
done
