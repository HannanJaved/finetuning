#!/bin/bash
#SBATCH --job-name=smoketest-DPO-UltraFeedback-Qwen3-4B-alpha
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/DPO-AO-UltraFeedback-DDP/smoketest-alpha/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/4B/DPO-AO-UltraFeedback-DDP/smoketest-alpha/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --partition=alpha
#SBATCH --exclusive

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-post-training/bin/activate

export HF_HOME="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
source /data/horse/ws/hama901h-Post-training/cache.sh
export PYTHONPATH="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/src:/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook:/data/horse/ws/hama901h-BFTranslation/venv-post-training/lib/python3.11/site-packages"

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

# Wandb settings - smoke test only, don't pollute the real project
export WANDB_MODE=disabled
export WANDB_NAME=smoketest-DPO-UltraFeedback-Qwen3-4B-alpha

cd /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/
ACCELERATE_CONFIG_FILE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B/zero2.yaml
CONFIG_FILE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/4B/DPO_AO_UltraFeedback/dpo_beta0.01_LR1e-6.yaml
SMOKETEST_OUTPUT_DIR=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache/smoketest-outputs/DPO-UltraFeedback-Qwen3-4B-alpha

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

# Same config as the real sweep (same model, dataset, batch size, max_length, etc.) but
# capped to a handful of steps, redirected to a scratch output dir, and with saving/eval/
# reporting disabled - this only exists to check the job starts, loads the model+dataset,
# and gets through a few train steps without hitting GPU OOM on the alpha A100 40GB nodes.
#LAUNCHERS
export CMD="scripts/dpo.py --config $CONFIG_FILE \
    --max_steps 5 \
    --save_strategy no \
    --logging_steps 1 \
    --report_to none \
    --output_dir $SMOKETEST_OUTPUT_DIR \
    --overwrite_output_dir"

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
