#!/bin/bash
#SBATCH --job-name=dpo_norm_tulu3_paper_actual_linear
#SBATCH --output=/data/cat/ws/hama901h-RL/.logs/Tulu3_DPO_NORM/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-RL/.logs/Tulu3_DPO_NORM/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA/12.4.0
source /data/cat/ws/hama901h-RL/.trl_venv/bin/activate

export HF_HOME="/data/cat/ws/hama901h-RL/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-RL/.cache"
# export PYTHONPATH="/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"
export PYTHONPATH="/data/cat/ws/hama901h-RL/.trl_venv/lib/python3.11/site-packages"
#pip show transformers

# Get master node hostname for distributed training
export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'
# try limited membership instead of full
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

#Distributed variables
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

cd /data/cat/ws/hama901h-RL/alignment-handbook/
ACCELERATE_CONFIG_FILE=/data/cat/ws/hama901h-RL/hpopt/zero3.yaml
CONFIG_FILE=/data/cat/ws/hama901h-RL/hpopt/dpo_norm/config_tulu3_dpo.yaml

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

#LAUNCHERS
export CMD="scripts/dpo.py --config $CONFIG_FILE"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

    
export ACC_LAUNCHER="accelerate launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
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
