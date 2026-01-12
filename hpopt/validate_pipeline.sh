#!/bin/bash
#SBATCH --job-name=validate_checkpoints
#SBATCH --output=/data/cat/ws/hama901h-RL/.logs/validation/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-RL/.logs/validation/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

# =============================================================================
# Validation Script for SFT and DPO Checkpoints
# =============================================================================
# This script validates trained checkpoints by computing evaluation loss.
#
# Usage:
#   sbatch validate_pipeline.sh [PIPELINE_CONFIG_FILE]
#
# Or for manual validation of specific checkpoints:
#   sbatch validate_pipeline.sh --sft-checkpoint /path/to/sft --dpo-checkpoint /path/to/dpo
#
# The script will:
#   1. Load the SFT checkpoint and compute validation loss on SFT dataset
#   2. Load the DPO checkpoint and compute validation loss on DPO dataset
#   3. Save results to a JSON file
# =============================================================================

set -e

echo "=============================================="
echo "Checkpoint Validation Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Start Time: $(date)"
echo "=============================================="

# -----------------------------------------------------------------------------
# Configuration - Update these paths or pass via command line
# -----------------------------------------------------------------------------
PIPELINE_CONFIG="${1:-/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_6600.yaml}"

echo "Using pipeline config: $PIPELINE_CONFIG"

# Parse configuration from YAML
parse_yaml() {
    python3 << EOF
import yaml
import sys

with open("$PIPELINE_CONFIG", 'r') as f:
    config = yaml.safe_load(f)

# Model settings
print(f"TORCH_DTYPE={config['model']['torch_dtype']}")

# Output settings
print(f"OUTPUT_BASE_DIR={config['output']['base_dir']}")
print(f"SFT_SUBDIR={config['output']['sft_subdir']}")
print(f"DPO_SUBDIR={config['output']['dpo_subdir']}")

# SFT settings
sft = config['sft']
print(f"SFT_DATASET={sft['dataset_name']}")
print(f"SFT_MAX_LENGTH={sft['max_length']}")
print(f"SFT_BATCH_SIZE={sft.get('per_device_eval_batch_size', sft.get('per_device_train_batch_size', 1))}")
print(f"SFT_SEED={sft['seed']}")

# DPO settings
dpo = config['dpo']
print(f"DPO_DATASET={dpo['dataset_name']}")
print(f"DPO_MAX_LENGTH={dpo['max_length']}")
print(f"DPO_MAX_PROMPT_LENGTH={dpo['max_prompt_length']}")
print(f"DPO_BATCH_SIZE={dpo.get('per_device_eval_batch_size', dpo.get('per_device_train_batch_size', 1))}")
print(f"DPO_BETA={dpo['beta']}")
print(f"DPO_SEED={dpo['seed']}")

# Check for chat template
if 'chat_template' in sft:
    import base64
    encoded = base64.b64encode(sft['chat_template'].encode()).decode()
    print(f"SFT_CHAT_TEMPLATE_B64={encoded}")
else:
    print("SFT_CHAT_TEMPLATE_B64=")
EOF
}

# Source the parsed variables
eval "$(parse_yaml)"

# Construct checkpoint paths
SFT_CHECKPOINT="${OUTPUT_BASE_DIR}/${SFT_SUBDIR}"
DPO_CHECKPOINT="${OUTPUT_BASE_DIR}/${DPO_SUBDIR}"

echo ""
echo "Configuration:"
echo "  SFT Checkpoint: $SFT_CHECKPOINT"
echo "  DPO Checkpoint: $DPO_CHECKPOINT"
echo "  SFT Dataset: $SFT_DATASET"
echo "  DPO Dataset: $DPO_DATASET"

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

echo ""
echo "Distributed Setup:"
echo "  Head Node: $head_node"
echo "  Master Address: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo "  World Size: $WORLD_SIZE"

cd /data/cat/ws/hama901h-RL/alignment-handbook/
ACCELERATE_CONFIG_FILE=recipes/accelerate_configs/zero3.yaml

# Create output directory for validation results
VALIDATION_OUTPUT_DIR="${OUTPUT_BASE_DIR}/validation_results"
mkdir -p "$VALIDATION_OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Generate Validation Config Files
# -----------------------------------------------------------------------------
CONFIG_TMP_DIR="/data/cat/ws/hama901h-RL/.tmp_configs"
mkdir -p "$CONFIG_TMP_DIR"

# SFT Validation Config
SFT_VAL_CONFIG="${CONFIG_TMP_DIR}/sft_val_config_${SLURM_JOB_ID}.yaml"

# Handle chat template if present
if [ -n "$SFT_CHAT_TEMPLATE_B64" ]; then
    SFT_CHAT_TEMPLATE=$(echo "$SFT_CHAT_TEMPLATE_B64" | base64 -d)
    CHAT_TEMPLATE_LINE="chat_template: '$SFT_CHAT_TEMPLATE'"
else
    CHAT_TEMPLATE_LINE=""
fi

cat > "$SFT_VAL_CONFIG" << SFTVALEOF
# Auto-generated SFT validation config for job $SLURM_JOB_ID
mode: sft
model_name_or_path: $SFT_CHECKPOINT
torch_dtype: $TORCH_DTYPE

dataset_name:
  $SFT_DATASET

bf16: true
max_length: $SFT_MAX_LENGTH
per_device_eval_batch_size: $SFT_BATCH_SIZE
gradient_checkpointing: true
output_dir: $VALIDATION_OUTPUT_DIR
results_file: ${VALIDATION_OUTPUT_DIR}/sft_validation_results.json
val_split_ratio: 0.05
val_split_seed: 42
seed: $SFT_SEED
log_level: info
$CHAT_TEMPLATE_LINE
SFTVALEOF

echo ""
echo "Generated SFT validation config at: $SFT_VAL_CONFIG"

# DPO Validation Config
DPO_VAL_CONFIG="${CONFIG_TMP_DIR}/dpo_val_config_${SLURM_JOB_ID}.yaml"
cat > "$DPO_VAL_CONFIG" << DPOVALEOF
# Auto-generated DPO validation config for job $SLURM_JOB_ID
mode: dpo
model_name_or_path: $DPO_CHECKPOINT
torch_dtype: $TORCH_DTYPE

dataset_name:
  $DPO_DATASET

bf16: true
beta: $DPO_BETA
max_length: $DPO_MAX_LENGTH
max_prompt_length: $DPO_MAX_PROMPT_LENGTH
per_device_eval_batch_size: $DPO_BATCH_SIZE
gradient_checkpointing: true
output_dir: $VALIDATION_OUTPUT_DIR
results_file: ${VALIDATION_OUTPUT_DIR}/dpo_validation_results.json
val_split_ratio: 0.05
val_split_seed: 42
seed: $DPO_SEED
log_level: info
DPOVALEOF

echo "Generated DPO validation config at: $DPO_VAL_CONFIG"

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
# Validate SFT Checkpoint
# =============================================================================
echo ""
echo "=============================================="
echo "Validating SFT Checkpoint"
echo "Checkpoint: $SFT_CHECKPOINT"
echo "Start Time: $(date)"
echo "=============================================="

if [ -d "$SFT_CHECKPOINT" ] && [ -f "$SFT_CHECKPOINT/config.json" ]; then
    SFT_VAL_CMD="scripts/validate_checkpoint.py --config $SFT_VAL_CONFIG"
    
    srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $SFT_VAL_CMD"
    
    SFT_VAL_EXIT_CODE=$?
    echo "SFT Validation completed with exit code: $SFT_VAL_EXIT_CODE"
    
    if [ -f "${VALIDATION_OUTPUT_DIR}/sft_validation_results.json" ]; then
        echo "SFT Validation Results:"
        cat "${VALIDATION_OUTPUT_DIR}/sft_validation_results.json"
    fi
else
    echo "WARNING: SFT checkpoint not found at $SFT_CHECKPOINT"
    SFT_VAL_EXIT_CODE=1
fi

# =============================================================================
# Validate DPO Checkpoint
# =============================================================================
echo ""
echo "=============================================="
echo "Validating DPO Checkpoint"
echo "Checkpoint: $DPO_CHECKPOINT"
echo "Start Time: $(date)"
echo "=============================================="

# Refresh master port
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)

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

if [ -d "$DPO_CHECKPOINT" ] && [ -f "$DPO_CHECKPOINT/config.json" ]; then
    DPO_VAL_CMD="scripts/validate_checkpoint.py --config $DPO_VAL_CONFIG"
    
    srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $DPO_VAL_CMD"
    
    DPO_VAL_EXIT_CODE=$?
    echo "DPO Validation completed with exit code: $DPO_VAL_EXIT_CODE"
    
    if [ -f "${VALIDATION_OUTPUT_DIR}/dpo_validation_results.json" ]; then
        echo "DPO Validation Results:"
        cat "${VALIDATION_OUTPUT_DIR}/dpo_validation_results.json"
    fi
else
    echo "WARNING: DPO checkpoint not found at $DPO_CHECKPOINT"
    DPO_VAL_EXIT_CODE=1
fi

# =============================================================================
# Cleanup and Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Validation Summary"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "SFT Checkpoint: $SFT_CHECKPOINT"
echo "SFT Validation Exit Code: $SFT_VAL_EXIT_CODE"
echo "DPO Checkpoint: $DPO_CHECKPOINT"
echo "DPO Validation Exit Code: $DPO_VAL_EXIT_CODE"
echo "Results saved to: $VALIDATION_OUTPUT_DIR"
echo "End Time: $(date)"
echo "=============================================="

# Cleanup temp config files
rm -f "$SFT_VAL_CONFIG" "$DPO_VAL_CONFIG"

# Return non-zero if either validation failed
if [ $SFT_VAL_EXIT_CODE -ne 0 ] || [ $DPO_VAL_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0
