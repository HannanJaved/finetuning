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

# Function to generate a random float between two values
random_float() {
  awk -v min="$1" -v max="$2" -v seed="$(date +%s%N | cut -b1-13)" 'BEGIN { srand(seed); print min + (max-min) * rand() }'
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

  # Create SLURM script
  cat <<EOL > "$SCRIPT_FILE"
#!/bin/bash
#SBATCH --job-name=dpo_random_$i
#SBATCH --output=$LOG_DIR/dpo_random_$i_%j.out
#SBATCH --error=$LOG_DIR/dpo_random_$i_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

module load CUDA/12.4.0
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Posttraining/.cache"
export PYTHONPATH="/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"

export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

ACCELERATE_CONFIG_FILE=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/zero3.yaml
CONFIG_FILE=$CONFIG_FILE

srun --wait=60 --kill-on-bad-exit=1 bash -c "accelerate launch --config_file $ACCELERATE_CONFIG_FILE --num_machines $SLURM_NNODES --num_processes $WORLD_SIZE --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT scripts/dpo.py --config $CONFIG_FILE"
EOL

  # Submit the job
  sbatch "$SCRIPT_FILE"
done