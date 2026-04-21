#!/bin/bash
set -euo pipefail

# Submit a grid sweep over learning rates and DPO beta values using the
# provided job and config templates in this directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_TEMPLATE="${SCRIPT_DIR}/LR1e7_Beta0.1.sh"
CONFIG_TEMPLATE="${SCRIPT_DIR}/dpo_beta0.1_LR.yaml"

if [ ! -f "${JOB_TEMPLATE}" ]; then
  echo "Job template not found: ${JOB_TEMPLATE}" >&2
  exit 1
fi
if [ ! -f "${CONFIG_TEMPLATE}" ]; then
  echo "Config template not found: ${CONFIG_TEMPLATE}" >&2
  exit 1
fi

# LRs and betas to sweep (as requested)
LRS=(5e-5 1e-6 3e-6 5e-6)
BETAS=(0.05 0.1 0.3 0.5 1.0)

# Parse options
DRY_RUN=0
show_help() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run|-n]  # creates files but does not call sbatch

Options:
  --dry-run, -n   Do not submit jobs; just create files and print the sbatch commands
  --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=1
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

echo "Using job template: ${JOB_TEMPLATE}"
echo "Using config template: ${CONFIG_TEMPLATE}"
if [ "${DRY_RUN}" -eq 1 ]; then
  echo "DRY RUN: will not call sbatch; use without --dry-run to actually submit."
fi

for lr in "${LRS[@]}"; do
  for beta in "${BETAS[@]}"; do
    # Name each job/config uniquely
    # replace dots in beta for safer filenames if needed (0.05 -> 0.05)
    job_name="Qwen3-4B-DPO-SFT_5e-4-Beta${beta}_LR${lr}"
    config_path="${SCRIPT_DIR}/dpo_beta${beta}_LR${lr}.yaml"
    job_path="${SCRIPT_DIR}/${job_name}.sh"

    # Create config: update learning_rate, beta, and output_dir
    sed \
      -e "s/^learning_rate: .*/learning_rate: ${lr}/" \
      -e "s/^beta: .*/beta: ${beta}/" \
      -e "s#^output_dir: .*#output_dir: /data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-4B-Base/SFT-sweep/${job_name}/#" \
      "${CONFIG_TEMPLATE}" > "${config_path}"

    # Create job script from template. Prefer replacing an existing WANDB_NAME
    # if present, otherwise insert one after WANDB_ENTITY.
    if grep -q "^export WANDB_NAME=" "${JOB_TEMPLATE}"; then
      sed \
        -e "s/^export WANDB_NAME=.*/export WANDB_NAME=${job_name}/" \
        -e "s/^#SBATCH --job-name=.*/#SBATCH --job-name=${job_name}/" \
        -e "s#^CONFIG_FILE=.*#CONFIG_FILE=${config_path}#" \
        "${JOB_TEMPLATE}" > "${job_path}"
    else
      sed \
        -e "s/^#SBATCH --job-name=.*/#SBATCH --job-name=${job_name}/" \
        -e "/^export WANDB_ENTITY=/a export WANDB_NAME=${job_name}" \
        -e "s#^CONFIG_FILE=.*#CONFIG_FILE=${config_path}#" \
        "${JOB_TEMPLATE}" > "${job_path}"
    fi

    chmod +x "${job_path}"
    if [ "${DRY_RUN}" -eq 1 ]; then
      echo "DRY RUN: would submit: sbatch ${job_path}  (config: ${config_path})"
    else
      echo "Submitting ${job_path} with ${config_path}"
      sbatch "${job_path}"
    fi
  done
done

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "Dry run finished: created files only."
else
  echo "All jobs submitted."
fi
