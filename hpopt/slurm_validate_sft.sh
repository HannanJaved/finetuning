#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

# =============================================================================
# SFT Validation SLURM Job
# =============================================================================
# Usage: sbatch slurm_validate_sft.sh <CONFIG_FILE>
# =============================================================================

set -e

CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not provided"
    echo "Usage: sbatch slurm_validate_sft.sh <CONFIG_FILE>"
    exit 1
fi

echo "=============================================="
echo "SFT Validation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Start Time: $(date)"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-RL/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-RL/.cache"
export PYTHONPATH="/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"

# NCCL settings
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
export TORCH_DISTRIBUTED_TIMEOUT=600
export TORCHELASTIC_MAX_FAILED_CONNECTIONS=60
export TORCH_DISTRIBUTED_HEARTBEAT_TIMEOUT=600
export TORCH_DISTRIBUTED_COODINATOR_TIMEOUT=600
export OMP_NUM_THREADS=18

# Distributed variables
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
export RDZV_HOST=$head_node
export RDZV_PORT=29400

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

echo "Distributed Setup:"
echo "  Head Node: $head_node"
echo "  Master Address: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo "  World Size: $WORLD_SIZE"

cd /data/cat/ws/hama901h-RL/alignment-handbook/
ACCELERATE_CONFIG_FILE=recipes/accelerate_configs/zero3.yaml

# -----------------------------------------------------------------------------
# Run Validation
# -----------------------------------------------------------------------------
SRUN_ARGS="--wait=60 --kill-on-bad-exit=1"

ACC_LAUNCHER="accelerate launch \
    --rdzv_conf rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3"

CMD="scripts/validate_sft.py --config $CONFIG_FILE"

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "SFT Validation Completed"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_CODE
