#!/bin/bash
#SBATCH --job-name=x_15450_1066_pipeline
#SBATCH --output=/data/cat/ws/hama901h-RL/.logs/optimal-N/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-RL/.logs/optimal-N/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

# =============================================================================
# SFT + DPO Training Pipeline
# =============================================================================
# This script runs a two-phase training pipeline:
#   Phase 1: Supervised Fine-Tuning (SFT) for X steps
#   Phase 2: Direct Preference Optimization (DPO) for (N-X) steps
#
# Usage:
#   sbatch sft_dpo_pipeline.sh [CONFIG_FILE]
#
# The config file should contain all pipeline parameters.
# Default config: /data/cat/ws/hama901h-RL/hpopt/config_sft_dpo_pipeline.yaml
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "SFT + DPO Pipeline Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Start Time: $(date)"
echo "=============================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PIPELINE_CONFIG="${1:-/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_15450.yaml}"
echo "Using pipeline config: $PIPELINE_CONFIG"

# Parse pipeline parameters from YAML config using Python
parse_yaml() {
    python3 << EOF
import yaml
import sys

with open("$PIPELINE_CONFIG", 'r') as f:
    config = yaml.safe_load(f)

# Pipeline settings
print(f"TOTAL_STEPS={config['pipeline']['total_steps']}")
print(f"SFT_STEPS={config['pipeline']['sft_steps']}")

# Model settings
print(f"BASE_MODEL_PATH={config['model']['base_model_path']}")
print(f"TORCH_DTYPE={config['model']['torch_dtype']}")

# Output settings
print(f"OUTPUT_BASE_DIR={config['output']['base_dir']}")
print(f"SFT_SUBDIR={config['output']['sft_subdir']}")
print(f"DPO_SUBDIR={config['output']['dpo_subdir']}")

# SFT settings
sft = config['sft']
print(f"SFT_DATASET={sft['dataset_name']}")
print(f"SFT_CHAT_TEMPLATE=\"{sft['chat_template']}\"")
print(f"SFT_LR={sft['learning_rate']}")
print(f"SFT_LR_SCHEDULER={sft['lr_scheduler_type']}")
print(f"SFT_WARMUP_RATIO={sft['warmup_ratio']}")
print(f"SFT_MAX_LENGTH={sft['max_length']}")
print(f"SFT_BATCH_SIZE={sft['per_device_train_batch_size']}")
print(f"SFT_EVAL_BATCH_SIZE={sft['per_device_eval_batch_size']}")
print(f"SFT_GRAD_ACCUM={sft['gradient_accumulation_steps']}")
print(f"SFT_BF16={str(sft['bf16']).lower()}")
print(f"SFT_GRAD_CKPT={str(sft['gradient_checkpointing']).lower()}")
print(f"SFT_LOG_STEPS={sft['logging_steps']}")
print(f"SFT_SAVE_STEPS={sft['save_steps']}")
print(f"SFT_SAVE_LIMIT={sft['save_total_limit']}")
print(f"SFT_EVAL_STRATEGY={sft['eval_strategy']}")
print(f"SFT_SEED={sft['seed']}")

# DPO settings
dpo = config['dpo']
print(f"DPO_DATASET={dpo['dataset_name']}")
print(f"DPO_BETA={dpo['beta']}")
print(f"DPO_LR={dpo['learning_rate']}")
print(f"DPO_LR_SCHEDULER={dpo['lr_scheduler_type']}")
print(f"DPO_WARMUP_RATIO={dpo['warmup_ratio']}")
print(f"DPO_MAX_LENGTH={dpo['max_length']}")
print(f"DPO_MAX_PROMPT_LENGTH={dpo['max_prompt_length']}")
print(f"DPO_BATCH_SIZE={dpo['per_device_train_batch_size']}")
print(f"DPO_EVAL_BATCH_SIZE={dpo['per_device_eval_batch_size']}")
print(f"DPO_GRAD_ACCUM={dpo['gradient_accumulation_steps']}")
print(f"DPO_BF16={str(dpo['bf16']).lower()}")
print(f"DPO_GRAD_CKPT={str(dpo['gradient_checkpointing']).lower()}")
print(f"DPO_LOG_STEPS={dpo['logging_steps']}")
print(f"DPO_SAVE_STEPS={dpo['save_steps']}")
print(f"DPO_SAVE_LIMIT={dpo['save_total_limit']}")
print(f"DPO_DO_EVAL={str(dpo['do_eval']).lower()}")
print(f"DPO_OPTIM={dpo['optim']}")
print(f"DPO_SEED={dpo['seed']}")

# Wandb settings
wandb = config['wandb']
print(f"WANDB_PROJECT_NAME={wandb['project']}")
print(f"WANDB_ENTITY_NAME={wandb['entity']}")
EOF
}

# Source the parsed variables
eval "$(parse_yaml)"

# Calculate DPO steps
DPO_STEPS=$((TOTAL_STEPS - SFT_STEPS))

echo "Pipeline Parameters:"
echo "  Total Steps: $TOTAL_STEPS"
echo "  SFT Steps: $SFT_STEPS"
echo "  DPO Steps: $DPO_STEPS"
echo "  Base Model: $BASE_MODEL_PATH"
echo "  SFT Output: $OUTPUT_BASE_DIR/$SFT_SUBDIR"
echo "  DPO Output: $OUTPUT_BASE_DIR/$DPO_SUBDIR"

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-RL/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-RL/.cache"
export PYTHONPATH="/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"

# NCCL settings for distributed training
export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'
export NCCL_IB_PKEY=0x2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_DEBUG=INFO
export NCCL_IB_RETRY_CNT=10
export NCCL_MIN_NCHANNELS=11
export NCCL_TREE_THRESHOLD=4294967296

# Torch distributed settings
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

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

echo "Distributed Setup:"
echo "  Head Node: $head_node"
echo "  Master Address: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo "  World Size: $WORLD_SIZE"
echo "  Processes per Node: $NPROC_PER_NODE"

# Wandb settings
export WANDB_PROJECT=$WANDB_PROJECT_NAME
export WANDB_ENTITY=$WANDB_ENTITY_NAME

cd /data/cat/ws/hama901h-RL/alignment-handbook/
ACCELERATE_CONFIG_FILE=recipes/accelerate_configs/zero3.yaml

# -----------------------------------------------------------------------------
# Generate SFT Config File (use shared filesystem, not /tmp which is node-local)
# -----------------------------------------------------------------------------
CONFIG_TMP_DIR="/data/cat/ws/hama901h-RL/.tmp_configs"
mkdir -p "$CONFIG_TMP_DIR"
SFT_CONFIG_FILE="${CONFIG_TMP_DIR}/sft_config_${SLURM_JOB_ID}.yaml"
SFT_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SFT_SUBDIR}"

cat > "$SFT_CONFIG_FILE" << SFTEOF
# Auto-generated SFT config for pipeline job $SLURM_JOB_ID
model_name_or_path: $BASE_MODEL_PATH
torch_dtype: $TORCH_DTYPE

chat_template: "$SFT_CHAT_TEMPLATE"

dataset_name:
  $SFT_DATASET

bf16: $SFT_BF16
eval_strategy: '$SFT_EVAL_STRATEGY'
learning_rate: $SFT_LR
lr_scheduler_type: $SFT_LR_SCHEDULER
warmup_ratio: $SFT_WARMUP_RATIO
max_length: $SFT_MAX_LENGTH
max_steps: $SFT_STEPS
num_train_epochs: -1
log_level: info
logging_steps: $SFT_LOG_STEPS
logging_strategy: steps
output_dir: $SFT_OUTPUT_DIR
overwrite_output_dir: false
per_device_train_batch_size: $SFT_BATCH_SIZE
per_device_eval_batch_size: $SFT_EVAL_BATCH_SIZE
gradient_accumulation_steps: $SFT_GRAD_ACCUM
gradient_checkpointing: $SFT_GRAD_CKPT
remove_unused_columns: true
save_strategy: "steps"
save_steps: $SFT_SAVE_STEPS
save_total_limit: $SFT_SAVE_LIMIT
seed: $SFT_SEED
SFTEOF

echo ""
echo "Generated SFT config at: $SFT_CONFIG_FILE"

# -----------------------------------------------------------------------------
# Generate DPO Config File (use shared filesystem, not /tmp which is node-local)
# -----------------------------------------------------------------------------
DPO_CONFIG_FILE="${CONFIG_TMP_DIR}/dpo_config_${SLURM_JOB_ID}.yaml"
DPO_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${DPO_SUBDIR}"

cat > "$DPO_CONFIG_FILE" << DPOEOF
# Auto-generated DPO config for pipeline job $SLURM_JOB_ID
model_name_or_path: $SFT_OUTPUT_DIR
torch_dtype: $TORCH_DTYPE

dataset_name:
  $DPO_DATASET

bf16: $DPO_BF16
beta: $DPO_BETA
do_eval: $DPO_DO_EVAL
gradient_checkpointing_kwargs:
  use_reentrant: false
learning_rate: $DPO_LR
lr_scheduler_type: $DPO_LR_SCHEDULER
warmup_ratio: $DPO_WARMUP_RATIO
max_length: $DPO_MAX_LENGTH
max_prompt_length: $DPO_MAX_PROMPT_LENGTH
max_steps: $DPO_STEPS
num_train_epochs: -1
log_level: info
logging_steps: $DPO_LOG_STEPS
optim: $DPO_OPTIM
output_dir: $DPO_OUTPUT_DIR
overwrite_output_dir: false
per_device_train_batch_size: $DPO_BATCH_SIZE
per_device_eval_batch_size: $DPO_EVAL_BATCH_SIZE
gradient_accumulation_steps: $DPO_GRAD_ACCUM
gradient_checkpointing: $DPO_GRAD_CKPT
save_strategy: "steps"
save_steps: $DPO_SAVE_STEPS
save_total_limit: $DPO_SAVE_LIMIT
seed: $DPO_SEED
DPOEOF

echo "Generated DPO config at: $DPO_CONFIG_FILE"

# -----------------------------------------------------------------------------
# SRUN and Accelerate Launcher Setup
# -----------------------------------------------------------------------------
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

ACC_LAUNCHER="accelerate launch \
    --rdzv_conf rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "

# =============================================================================
# PHASE 1: SFT Training
# =============================================================================
echo ""
echo "=============================================="
echo "PHASE 1: SFT Training"
echo "Steps: $SFT_STEPS"
echo "Start Time: $(date)"
echo "=============================================="

SFT_CMD="scripts/sft.py --config $SFT_CONFIG_FILE"

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $SFT_CMD"

SFT_EXIT_CODE=$?
echo ""
echo "SFT Phase completed with exit code: $SFT_EXIT_CODE"
echo "SFT End Time: $(date)"

if [ $SFT_EXIT_CODE -ne 0 ]; then
    echo "ERROR: SFT training failed! Aborting pipeline."
    exit $SFT_EXIT_CODE
fi

# =============================================================================
# PHASE 2: DPO Training
# =============================================================================
echo ""
echo "=============================================="
echo "PHASE 2: DPO Training"
echo "Steps: $DPO_STEPS"
echo "Start Time: $(date)"
echo "=============================================="

# Refresh master port to avoid conflicts
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
echo "New Master Port for DPO: $MASTER_PORT"

# Update accelerate launcher with new port
ACC_LAUNCHER="accelerate launch \
    --rdzv_conf rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "

DPO_CMD="scripts/dpo.py --config $DPO_CONFIG_FILE"

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $DPO_CMD"

DPO_EXIT_CODE=$?
echo ""
echo "DPO Phase completed with exit code: $DPO_EXIT_CODE"
echo "DPO End Time: $(date)"

# =============================================================================
# Cleanup and Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Pipeline Summary"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Total Steps: $TOTAL_STEPS"
echo "SFT Steps: $SFT_STEPS (Exit Code: $SFT_EXIT_CODE)"
echo "DPO Steps: $DPO_STEPS (Exit Code: $DPO_EXIT_CODE)"
echo "SFT Model: $SFT_OUTPUT_DIR"
echo "DPO Model: $DPO_OUTPUT_DIR"
echo "End Time: $(date)"
echo "=============================================="

# Cleanup temp config files
rm -f "$SFT_CONFIG_FILE" "$DPO_CONFIG_FILE"

exit $DPO_EXIT_CODE
