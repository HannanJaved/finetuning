#!/bin/bash
#SBATCH --job-name=batch_validate
#SBATCH --output=/data/cat/ws/hama901h-RL/.logs/validation/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-RL/.logs/validation/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

# =============================================================================
# Batch Validation Script for Multiple SFT/DPO Checkpoints
# =============================================================================
# This script validates multiple trained checkpoints from the Llama-8B-optimal-N
# experiments by computing evaluation loss.
#
# Usage:
#   sbatch batch_validate_checkpoints.sh
#
# The script will iterate through all config files and validate corresponding
# checkpoints, saving all results to a combined JSON file.
# =============================================================================

set -e

echo "=============================================="
echo "Batch Checkpoint Validation Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
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
export TORCH_DISTRIBUTED_TIMEOUT=300
export OMP_NUM_THREADS=18

# Distributed variables
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
export RDZV_HOST=$head_node

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

cd /data/cat/ws/hama901h-RL/alignment-handbook/
ACCELERATE_CONFIG_FILE=recipes/accelerate_configs/zero3.yaml

# Results directory
RESULTS_DIR="/data/cat/ws/hama901h-RL/evaluation_results/validation"
mkdir -p "$RESULTS_DIR"

CONFIG_TMP_DIR="/data/cat/ws/hama901h-RL/.tmp_configs"
mkdir -p "$CONFIG_TMP_DIR"

# Combined results file
COMBINED_RESULTS="${RESULTS_DIR}/all_validation_results.json"
echo '{"experiments": []}' > "$COMBINED_RESULTS"

# -----------------------------------------------------------------------------
# List of pipeline configs to validate
# -----------------------------------------------------------------------------
# Add your pipeline config files here
PIPELINE_CONFIGS=(
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_6600.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_8258.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_9100.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_10300.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_11150.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_12250.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_13300.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_15450.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline_x_16516.yaml"
    "/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N/config_sft_dpo_pipeline.yaml"
)

# Function to validate a single config
validate_config() {
    local PIPELINE_CONFIG="$1"
    local CONFIG_NAME=$(basename "$PIPELINE_CONFIG" .yaml)
    
    echo ""
    echo "=============================================="
    echo "Processing: $CONFIG_NAME"
    echo "Config: $PIPELINE_CONFIG"
    echo "=============================================="
    
    # Parse YAML config
    eval "$(python3 << EOF
import yaml
with open("$PIPELINE_CONFIG", 'r') as f:
    config = yaml.safe_load(f)

print(f"TORCH_DTYPE={config['model']['torch_dtype']}")
print(f"OUTPUT_BASE_DIR={config['output']['base_dir']}")
print(f"SFT_SUBDIR={config['output']['sft_subdir']}")
print(f"DPO_SUBDIR={config['output']['dpo_subdir']}")

sft = config['sft']
print(f"SFT_DATASET={sft['dataset_name']}")
print(f"SFT_MAX_LENGTH={sft['max_length']}")
print(f"SFT_BATCH_SIZE={sft.get('per_device_eval_batch_size', sft.get('per_device_train_batch_size', 1))}")
print(f"SFT_SEED={sft['seed']}")

dpo = config['dpo']
print(f"DPO_DATASET={dpo['dataset_name']}")
print(f"DPO_MAX_LENGTH={dpo['max_length']}")
print(f"DPO_MAX_PROMPT_LENGTH={dpo['max_prompt_length']}")
print(f"DPO_BATCH_SIZE={dpo.get('per_device_eval_batch_size', dpo.get('per_device_train_batch_size', 1))}")
print(f"DPO_BETA={dpo['beta']}")
print(f"DPO_SEED={dpo['seed']}")

pipeline = config['pipeline']
print(f"TOTAL_STEPS={pipeline['total_steps']}")
print(f"SFT_STEPS={pipeline['sft_steps']}")
EOF
)"
    
    SFT_CHECKPOINT="${OUTPUT_BASE_DIR}/${SFT_SUBDIR}"
    DPO_CHECKPOINT="${OUTPUT_BASE_DIR}/${DPO_SUBDIR}"
    
    EXPERIMENT_RESULTS_DIR="${RESULTS_DIR}/${CONFIG_NAME}"
    mkdir -p "$EXPERIMENT_RESULTS_DIR"
    
    # Refresh master port
    export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
    
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
    
    SFT_VAL_EXIT_CODE=0
    DPO_VAL_EXIT_CODE=0
    
    # Validate SFT
    if [ -d "$SFT_CHECKPOINT" ] && [ -f "$SFT_CHECKPOINT/config.json" ]; then
        echo "Validating SFT checkpoint: $SFT_CHECKPOINT"
        
        SFT_VAL_CONFIG="${CONFIG_TMP_DIR}/sft_val_${CONFIG_NAME}_${SLURM_JOB_ID}.yaml"
        cat > "$SFT_VAL_CONFIG" << SFTEOF
mode: sft
model_name_or_path: $SFT_CHECKPOINT
torch_dtype: $TORCH_DTYPE
dataset_name:
  $SFT_DATASET
bf16: true
max_length: $SFT_MAX_LENGTH
per_device_eval_batch_size: $SFT_BATCH_SIZE
gradient_checkpointing: true
output_dir: $EXPERIMENT_RESULTS_DIR
results_file: ${EXPERIMENT_RESULTS_DIR}/sft_validation_results.json
val_split_ratio: 0.05
val_split_seed: 42
seed: $SFT_SEED
log_level: info
SFTEOF
        
        SFT_VAL_CMD="scripts/validate_checkpoint.py --config $SFT_VAL_CONFIG"
        srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $SFT_VAL_CMD" || SFT_VAL_EXIT_CODE=$?
        
        rm -f "$SFT_VAL_CONFIG"
    else
        echo "WARNING: SFT checkpoint not found at $SFT_CHECKPOINT"
        SFT_VAL_EXIT_CODE=1
    fi
    
    # Refresh port for DPO
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
        --tee 3"
    
    # Validate DPO
    if [ -d "$DPO_CHECKPOINT" ] && [ -f "$DPO_CHECKPOINT/config.json" ]; then
        echo "Validating DPO checkpoint: $DPO_CHECKPOINT"
        
        DPO_VAL_CONFIG="${CONFIG_TMP_DIR}/dpo_val_${CONFIG_NAME}_${SLURM_JOB_ID}.yaml"
        cat > "$DPO_VAL_CONFIG" << DPOEOF
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
output_dir: $EXPERIMENT_RESULTS_DIR
results_file: ${EXPERIMENT_RESULTS_DIR}/dpo_validation_results.json
val_split_ratio: 0.05
val_split_seed: 42
seed: $DPO_SEED
log_level: info
DPOEOF
        
        DPO_VAL_CMD="scripts/validate_checkpoint.py --config $DPO_VAL_CONFIG"
        srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $DPO_VAL_CMD" || DPO_VAL_EXIT_CODE=$?
        
        rm -f "$DPO_VAL_CONFIG"
    else
        echo "WARNING: DPO checkpoint not found at $DPO_CHECKPOINT"
        DPO_VAL_EXIT_CODE=1
    fi
    
    echo "Finished: $CONFIG_NAME (SFT: $SFT_VAL_EXIT_CODE, DPO: $DPO_VAL_EXIT_CODE)"
}

# -----------------------------------------------------------------------------
# Main validation loop
# -----------------------------------------------------------------------------
for config in "${PIPELINE_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        validate_config "$config"
    else
        echo "WARNING: Config file not found: $config"
    fi
done

# -----------------------------------------------------------------------------
# Combine all results
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Combining all validation results..."
echo "=============================================="

python3 << PYEOF
import json
import os
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
combined = {"experiments": []}

for exp_dir in sorted(results_dir.iterdir()):
    if exp_dir.is_dir() and exp_dir.name != "validation":
        exp_result = {
            "name": exp_dir.name,
            "sft": None,
            "dpo": None
        }
        
        sft_file = exp_dir / "sft_validation_results.json"
        if sft_file.exists():
            with open(sft_file) as f:
                exp_result["sft"] = json.load(f)
        
        dpo_file = exp_dir / "dpo_validation_results.json"
        if dpo_file.exists():
            with open(dpo_file) as f:
                exp_result["dpo"] = json.load(f)
        
        if exp_result["sft"] or exp_result["dpo"]:
            combined["experiments"].append(exp_result)

with open("$COMBINED_RESULTS", "w") as f:
    json.dump(combined, f, indent=2)

print(f"Combined results saved to: $COMBINED_RESULTS")
print(f"Total experiments: {len(combined['experiments'])}")
PYEOF

echo ""
echo "=============================================="
echo "Batch Validation Summary"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Total configs processed: ${#PIPELINE_CONFIGS[@]}"
echo "Results directory: $RESULTS_DIR"
echo "Combined results: $COMBINED_RESULTS"
echo "End Time: $(date)"
echo "=============================================="
