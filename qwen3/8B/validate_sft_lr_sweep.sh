#!/bin/bash
#SBATCH --job-name=Qwen3-8B-SFT-VAL
#SBATCH --output=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/8B/%x_%A_%a.out
#SBATCH --error=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/8B/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=capella
#SBATCH --array=0-4

set -euo pipefail

echo "JOB NAME" $SLURM_JOB_NAME

module load release/24.10
module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export PYTHONPATH="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook:/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"

# Get master node hostname for distributed validation
export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'
export NCCL_IB_PKEY=0x2

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_DEBUG=INFO
export NCCL_IB_RETRY_CNT=10
export NCCL_MIN_NCHANNELS=11
export NCCL_TREE_THRESHOLD=4294967296
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_TIMEOUT=300
export TORCHELASTIC_MAX_FAILED_CONNECTIONS=60
export TORCH_DISTRIBUTED_HEARTBEAT_TIMEOUT=300
export TORCH_DISTRIBUTED_COODINATOR_TIMEOUT=300
export OMP_NUM_THREADS=18

# Distributed variables
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

export RDZV_HOST=$head_node
export RDZV_PORT=29400

echo "head_node=$head_node"

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

echo NPROC_PER_NODE=$NPROC_PER_NODE

# Wandb settings
export WANDB_PROJECT=instruction-tuning
export WANDB_ENTITY=openeurollm-project

BASE_DIR="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/8B"
LRS=(5e-4 1e-5 3e-5 5e-5 1e-6)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
export WANDB_NAME="Qwen3-8B-SFT-VAL-LR${LR}"

cd /data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/
ACCELERATE_CONFIG_FILE=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/zero3.yaml
CONFIG_FILE="${BASE_DIR}/config_olmo3_sft_${LR}.yaml"

TRAIN_OUTPUT_DIR=$(grep -E '^output_dir:' "$CONFIG_FILE" | awk '{print $2}')
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR%/}"
LATEST_CHECKPOINT=$(ls -d "${TRAIN_OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
  MODEL_PATH="${LATEST_CHECKPOINT}"
else
  MODEL_PATH="${TRAIN_OUTPUT_DIR}"
fi
VALIDATION_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}/validation"

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
echo "MODEL" $MODEL_PATH
echo "VALIDATION_OUTPUT" $VALIDATION_OUTPUT_DIR
pwd -P

# LAUNCHERS
export CMD="scripts/validate_sft.py --config $CONFIG_FILE --model_name_or_path $MODEL_PATH --output_dir $VALIDATION_OUTPUT_DIR"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

export ACC_LAUNCHER="accelerate launch \
    --rdzv_conf \"rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
