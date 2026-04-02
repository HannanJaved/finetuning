#!/bin/bash
set -euo pipefail

BASE_DIR="/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/0.6B"
JOB_TEMPLATE="${BASE_DIR}/sft_olmo3_1e5.sh"
CONFIG_TEMPLATE="${BASE_DIR}/config_olmo3_sft_1e5.yaml"

LRS=(5e-4 1e-5 3e-5 5e-5 1e-6)

for lr in "${LRS[@]}"; do
  job_name="Qwen3-0.6B-SFT-LR${lr}"
  config_path="${BASE_DIR}/config_olmo3_sft_${lr}.yaml"
  job_path="${BASE_DIR}/sft_olmo3_${lr}.sh"

  sed \
    -e "s/^learning_rate: .*/learning_rate: ${lr}/" \
    -e "s#^output_dir: .*#output_dir: /data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-0.6B-Base/SFT-sweep/${job_name}/#" \
    "${CONFIG_TEMPLATE}" > "${config_path}"

  sed \
    -e "s/^#SBATCH --job-name=.*/#SBATCH --job-name=${job_name}/" \
    -e "/^export WANDB_ENTITY=/a export WANDB_NAME=${job_name}" \
    -e "s#^CONFIG_FILE=.*#CONFIG_FILE=${config_path}#" \
    "${JOB_TEMPLATE}" > "${job_path}"

  chmod +x "${job_path}"
  echo "Submitting ${job_path} with ${config_path}"
  sbatch "${job_path}"
done
