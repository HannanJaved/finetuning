#!/bin/bash
set -euo pipefail

# Generate validation YAML configs and SLURM job scripts for all 16 DPO checkpoints
# of the 1B model, then optionally submit them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_TEMPLATE="${SCRIPT_DIR}/validate_dpo_template_1B.yaml"

CHECKPOINT_BASE="/data/horse/ws/hama901h-BFTranslation/checkpoints/Gemma3/gemma-3-1b-pt/SFT-sweep/gemma-3-1b-pt-SFT-LR3e-5/DPO"
LOG_DIR="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Gemma3/1B/DPO/validate"
AH_DIR="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook"
ACCELERATE_CONFIG="${AH_DIR}/recipes/accelerate_configs/ddp.yaml"

LRS=(1e-5 5e-6 3e-6 1e-6)
BETAS=(0.01 0.05 0.1 0.3)

DRY_RUN=0
show_help() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run|-n]

Options:
  --dry-run, -n   Create files but do not call sbatch
  --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n) DRY_RUN=1; shift ;;
    --help) show_help; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; show_help; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}"

echo "Generating validation jobs for Gemma3-1B (16 configs)"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN: will not call sbatch"
fi

for beta in "${BETAS[@]}"; do
  for lr in "${LRS[@]}"; do
    job_name="Val-Gemma3-1B-DPO-Beta${beta}_LR${lr}"
    checkpoint_dir="${CHECKPOINT_BASE}/Beta${beta}_LR${lr}"
    config_path="${SCRIPT_DIR}/validate_dpo_beta${beta}_LR${lr}_1B.yaml"
    job_path="${SCRIPT_DIR}/${job_name}.sh"

    if [[ ! -d "${checkpoint_dir}" ]]; then
      echo "WARNING: checkpoint not found, skipping: ${checkpoint_dir}"
      continue
    fi

    # Create validation config
    sed \
      -e "s|CHECKPOINT_PATH|${checkpoint_dir}|" \
      -e "s|BETA_VALUE|${beta}|" \
      -e "s|RUN_NAME|Beta${beta}_LR${lr}|" \
      "${CONFIG_TEMPLATE}" > "${config_path}"

    # Create SLURM job script
    cat > "${job_path}" <<SLURM
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=capella

echo "JOB NAME" \$SLURM_JOB_NAME

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-post-training/bin/activate

export HF_HOME="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
source /data/horse/ws/hama901h-Post-training/cache.sh
export PYTHONPATH="${AH_DIR}/src:${AH_DIR}:/data/horse/ws/hama901h-BFTranslation/venv-post-training/lib/python3.11/site-packages"

export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'
export NCCL_IB_PKEY=0x2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_DEBUG=INFO
export NCCL_IB_RETRY_CNT=10
export NCCL_MIN_NCHANNELS=11
export NCCL_TREE_THRESHOLD=4294967296
export OMP_NUM_THREADS=18

export MASTER_PORT=\$(shuf -i 20000-29999 -n 1)
export MASTER_ADDR=\$(hostname)
export WORLD_SIZE=\$(nvidia-smi -L | wc -l)

export WANDB_DISABLED=true

cd ${AH_DIR}
CONFIG_FILE=${config_path}

echo "CONFIG" \$CONFIG_FILE

srun --wait=60 --kill-on-bad-exit=1 --jobid \$SLURM_JOB_ID bash -c "accelerate launch \\
    --config_file ${ACCELERATE_CONFIG} \\
    --num_machines 1 \\
    --num_processes \$WORLD_SIZE \\
    --main_process_ip \$MASTER_ADDR \\
    --main_process_port \$MASTER_PORT \\
    scripts/validate_dpo.py --config \$CONFIG_FILE"

echo "END TIME: \$(date)"
SLURM

    chmod +x "${job_path}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "DRY RUN: sbatch ${job_path}  (config: ${config_path})"
    else
      echo "Submitting ${job_path}"
      sbatch "${job_path}"
    fi
  done
done

echo "Done."
