#!/bin/bash
# Training body for the DPO language-mix jobs.
# Submit a dpo-enXX.sh wrapper with sbatch. Do not sbatch this file.
set -euo pipefail

CONFIG="${1:?usage: launch_dpo.sh /path/to/dpo-enXX.yaml}"
REPO="/gpfs/projects/ehpc533/users/hannan/post-training"
DATA_ROOT="/gpfs/projects/ehpc533/users/hannan/data/Dolci-Instruct-DPO-translated"
VENV="${REPO}/.venv"
UV_BIN="/gpfs/projects/ehpc533/users/hannan/.local/bin"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

cd "${REPO}"

# Login-node env.sh sets a socks proxy that compute nodes cannot use.
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy || true

export PATH="${VENV}/bin:${UV_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-80}"

export HF_HOME="/gpfs/projects/ehpc533/users/hannan/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export WANDB_DIR="/gpfs/projects/ehpc533/users/hannan/finetuning/OELLM/DPO-datamix-prelude/outputs"
export WANDB_PROJECT="oellm-dpo-datamix"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${WANDB_DIR}" "${TMPDIR}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-120}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

if [[ ! -x "${VENV}/bin/accelerate" ]]; then
  echo "No training environment at ${VENV}." >&2
  echo "On a login node:" >&2
  echo "  export PATH=${UV_BIN}:\$PATH" >&2
  echo "  cd ${REPO}" >&2
  echo "  uv sync --extra trl" >&2
  exit 1
fi

require_real_file() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo "Missing data file: ${f}" >&2
    exit 1
  fi
  if head -c 64 "${f}" | grep -q "git-lfs"; then
    echo "Still a Git LFS pointer: ${f}" >&2
    echo "On a login node:" >&2
    echo "  module load git-lfs/3.4.1" >&2
    echo "  cd ${DATA_ROOT}" >&2
    echo "  git lfs pull" >&2
    exit 1
  fi
}

require_real_file "${DATA_ROOT}/en/shard00-of-25.parquet"
if grep -q "translated_dpo.jsonl" "${CONFIG}"; then
  for lang in cs de el es fi fr it pl ro sv uk; do
    require_real_file "${DATA_ROOT}/${lang}/translated_dpo.jsonl"
  done
fi

HOST0="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_ADDR="$(getent hosts "${HOST0}" | awk '{print $1; exit}' || true)"
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="${HOST0}"
fi
export MASTER_ADDR
export MASTER_PORT="$((29500 + ${SLURM_JOB_ID} % 2000))"
export NUM_NODES="${SLURM_NNODES}"
export GPUS_PER_NODE=4
export WORLD_SIZE="$((NUM_NODES * GPUS_PER_NODE))"

echo "=========================================="
echo "Job ${SLURM_JOB_ID} started $(date)"
echo "Config: ${CONFIG}"
echo "Nodes: ${NUM_NODES} x ${GPUS_PER_NODE} GPUs (WORLD_SIZE=${WORLD_SIZE})"
echo "MASTER_ADDR:MASTER_PORT=${MASTER_ADDR}:${MASTER_PORT}"
echo "=========================================="
nvidia-smi -L || true

handle_signal() {
  echo "[$(date)] SIGUSR1: wall time is close. Requeue so training resumes from the last checkpoint."
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap handle_signal USR1

LAUNCHER="accelerate launch \
  --num_machines ${NUM_NODES} \
  --num_processes ${WORLD_SIZE} \
  --machine_rank \$SLURM_PROCID \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --use_deepspeed \
  --deepspeed_multinode_launcher standard \
  --same_network \
  --rdzv_backend static \
  --max_restarts 0 \
  --role \$(hostname -s): \
  scripts/train.py --config \"${CONFIG}\""

set +e
srun --ntasks="${NUM_NODES}" --ntasks-per-node=1 --export=ALL --wait=60 --kill-on-bad-exit=1 \
  bash -c "${LAUNCHER}" &
wait $!
EXIT_CODE=$?
set -e

echo "Job ${SLURM_JOB_ID} finished $(date) with exit code ${EXIT_CODE}"
exit "${EXIT_CODE}"
