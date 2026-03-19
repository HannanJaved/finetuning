#!/bin/bash

# Define ranges for random search
LR_MIN=5e-8
LR_MAX=5e-6
BETA_MIN=0.5
BETA_MAX=5

# Number of random configurations to generate
NUM_SAMPLES=30

# Directories for configs and scripts
CONFIG_DIR="/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/RS/configs"
SCRIPT_DIR="/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/RS/scripts"
LOG_DIR="/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/RandomSearch"

mkdir -p "$CONFIG_DIR" "$SCRIPT_DIR" "$LOG_DIR"

# Function to generate a random float between two values (uniform in linear space)
random_float() {
  python - "$1" "$2" <<'PY'
import random
import sys

min_val = float(sys.argv[1])
max_val = float(sys.argv[2])
print(min_val + (max_val - min_val) * random.random())
PY
}


# Generate random configurations and scripts
for i in $(seq 1 $NUM_SAMPLES); do
  LR=$(random_float $LR_MIN $LR_MAX)
  BETA=$(random_float $BETA_MIN $BETA_MAX)

  CONFIG_FILE="$CONFIG_DIR/config_dpo_random_$i.yaml"
  SCRIPT_FILE="$SCRIPT_DIR/dpo_random_$i.sh"

  # Create config file
  cat <<EOL > "$CONFIG_FILE"
# Model arguments
model_name_or_path: /data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-1.7B-Base/SFT/LR5e5/
torch_dtype: bfloat16

# Data training arguments
dataset_name:
  allenai/Dolci-Instruct-DPO

# DPOTrainer arguments
bf16: true
beta: $BETA
do_eval: false
gradient_checkpointing_kwargs:
  use_reentrant: False
learning_rate: $LR
log_level: info
logging_steps: 10
lr_scheduler_type: linear
max_length: 16384
max_prompt_length: 512
num_train_epochs: 1
optim: adamw_torch
output_dir: /data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-1.7B-Base/SFT/Lr5e5/DPO_NORM/RandomSearch/Beta${BETA}_LR${LR}/
overwrite_output_dir: true
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
save_strategy: "steps"
save_steps: 200
save_total_limit: 20
seed: 8
warmup_ratio: 0.1
length_normalize_logps: true
resume_from_checkpoint: false
EOL

  # Create SLURM script from a runtime-safe template.
  cat <<'EOL' > "$SCRIPT_FILE"
#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=__LOG_DIR__/__JOB_NAME___%j.out
#SBATCH --error=__LOG_DIR__/__JOB_NAME___%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

set -euo pipefail

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Posttraining/.cache"
export PYTHONPATH="/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"
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
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

export RDZV_HOST=$head_node
export RDZV_PORT=29400

echo "head_node=$head_node"

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
echo NPROC_PER_NODE=$NPROC_PER_NODE

export WANDB_PROJECT=instruction-tuning
export WANDB_ENTITY=openeurollm-project

cd /data/cat/ws/hama901h-Posttraining/finetuning/alignment-handbook/

ACCELERATE_CONFIG_FILE=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/zero3.yaml
CONFIG_FILE=__CONFIG_FILE__

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

export CMD="scripts/dpo.py --config $CONFIG_FILE"

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

echo "END TIME: $(date)"
echo "END $SLURM_JOBID: $(date)"
EOL

  sed -i \
    -e "s|__JOB_NAME__|dpo_random_$i|g" \
    -e "s|__LOG_DIR__|$LOG_DIR|g" \
    -e "s|__CONFIG_FILE__|$CONFIG_FILE|g" \
    "$SCRIPT_FILE"

  # Submit the job
  sbatch "$SCRIPT_FILE"
done