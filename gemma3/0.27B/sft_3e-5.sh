#!/bin/bash
#SBATCH --job-name=Gemma3-270m-SFT-LR3e-5
#SBATCH --output=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Gemma3/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Gemma3/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=capella

set -euo pipefail

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA/12.6.0
VENV=/data/horse/ws/hama901h-BFTranslation/venv-post-training
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:/data/horse/ws/hama901h-BFTranslation/libffi-install/lib64:${LD_LIBRARY_PATH:-}
source "$VENV/bin/activate"

export HF_HOME="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
source /data/cat/ws/hama901h-Post-training/cache.sh

# Get master node hostname for distributed training
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
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCHELASTIC_MAX_FAILED_CONNECTIONS=60
export TORCH_DISTRIBUTED_HEARTBEAT_TIMEOUT=7200
export TORCH_DISTRIBUTED_COODINATOR_TIMEOUT=7200
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
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
export WANDB_NAME=Gemma3-270m-SFT-LR3e-5

cd /data/cat/ws/hama901h-Post-training/hama901h-Posttraining/post-training
CONFIG_FILE=/data/cat/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/gemma3/0.27B/sft_3e-5.yaml

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

CMD="scripts/train.py --config $CONFIG_FILE"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

ACCELERATE_BIN="$VENV/bin/accelerate"
ACC_LAUNCHER="$ACCELERATE_BIN launch \
    --rdzv_conf \"rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\" \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): \
    --mixed_precision bf16 \
    --dynamo_backend inductor \
    --use_deepspeed \
    --deepspeed_multinode_launcher standard \
    --tee 3 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
