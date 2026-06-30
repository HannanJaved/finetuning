#!/bin/bash
# Shared launcher for datamix-9b strategy pilots.
# Required env:
#   PILOT_STRATEGY   e.g. zero2-liger, zero2-bs2
#   ACCELERATE_CONFIG_FILE
# Optional env:
#   PILOT_CONFIG     defaults to config_pilot.yaml

set -euo pipefail

: "${PILOT_STRATEGY:?PILOT_STRATEGY must be set}"
: "${ACCELERATE_CONFIG_FILE:?ACCELERATE_CONFIG_FILE must be set}"

PILOT_DIR=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B
AH_DIR=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook
BASE_CONFIG=${PILOT_CONFIG:-${PILOT_DIR}/config_pilot.yaml}
RUN_CONFIG_DIR=${PILOT_DIR}/pilot_configs
RUN_CONFIG=${RUN_CONFIG_DIR}/datamix-9b-pilot-${PILOT_STRATEGY}-${SLURM_JOB_ID:-local}.yaml
mkdir -p "$RUN_CONFIG_DIR"

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-post-training/bin/activate

export HF_HOME="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
source /data/horse/ws/hama901h-Post-training/cache.sh
export PYTHONPATH="${AH_DIR}/src:${AH_DIR}:/data/horse/ws/hama901h-BFTranslation/venv-post-training/lib/python3.11/site-packages"

export WANDB_DISABLED=true
export WANDB_MODE=disabled

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

export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE * SLURM_NNODES))

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
export RDZV_HOST=$head_node
export RDZV_PORT=29400

sed "s|PLACEHOLDER|${PILOT_STRATEGY}|" "$BASE_CONFIG" > "$RUN_CONFIG"

echo "PILOT_STRATEGY=$PILOT_STRATEGY"
echo "ACCELERATE_CONFIG=$ACCELERATE_CONFIG_FILE"
echo "CONFIG=$RUN_CONFIG"
echo "head_node=$head_node"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NPROC_PER_NODE=$(nvidia-smi -L | wc -l)"

cd "$AH_DIR"
export CMD="scripts/sft.py --config $RUN_CONFIG"

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
