#!/bin/bash
# =============================================================================
# Submit Validation Jobs for All Pipeline Configs
# =============================================================================
# This script submits 2 SLURM jobs per pipeline config:
#   - One job for SFT validation
#   - One job for DPO validation
#
# Usage:
#   ./submit_validation_jobs.sh [CONFIG_DIR]
#
# Default CONFIG_DIR: /data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N
# =============================================================================

set -e

CONFIG_DIR="${1:-/data/cat/ws/hama901h-RL/hpopt/Llama-8B-optimal-N}"
RESULTS_BASE_DIR="/data/cat/ws/hama901h-RL/evaluation_results/validation"
LOG_DIR="/data/cat/ws/hama901h-RL/.logs/validation"
TMP_CONFIG_DIR="/data/cat/ws/hama901h-RL/.tmp_configs/validation"

mkdir -p "$RESULTS_BASE_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$TMP_CONFIG_DIR"

echo "=============================================="
echo "Submitting Validation Jobs"
echo "Config directory: $CONFIG_DIR"
echo "Results directory: $RESULTS_BASE_DIR"
echo "=============================================="

# Track submitted jobs
SUBMITTED_JOBS=()

# Process each pipeline config
for config_file in "$CONFIG_DIR"/config_sft_dpo_pipeline*.yaml; do
    if [ ! -f "$config_file" ]; then
        continue
    fi
    
    CONFIG_NAME=$(basename "$config_file" .yaml)
    echo ""
    echo "Processing: $CONFIG_NAME"
    
    # Parse the YAML config to get paths
    eval "$(python3 << EOF
import yaml
with open("$config_file", 'r') as f:
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
EOF
)"
    
    SFT_CHECKPOINT="${OUTPUT_BASE_DIR}/${SFT_SUBDIR}"
    DPO_CHECKPOINT="${OUTPUT_BASE_DIR}/${DPO_SUBDIR}"
    
    EXPERIMENT_RESULTS_DIR="${RESULTS_BASE_DIR}/${CONFIG_NAME}"
    mkdir -p "$EXPERIMENT_RESULTS_DIR"
    
    # -------------------------------------------------------------------------
    # Submit SFT Validation Job
    # -------------------------------------------------------------------------
    if [ -d "$SFT_CHECKPOINT" ] && [ -f "$SFT_CHECKPOINT/config.json" ]; then
        SFT_VAL_CONFIG="${TMP_CONFIG_DIR}/sft_val_${CONFIG_NAME}.yaml"
        cat > "$SFT_VAL_CONFIG" << SFTEOF
# Auto-generated SFT validation config
model_name_or_path: $SFT_CHECKPOINT
torch_dtype: $TORCH_DTYPE

dataset_name:
  $SFT_DATASET

bf16: true
max_length: $SFT_MAX_LENGTH
per_device_eval_batch_size: $SFT_BATCH_SIZE
gradient_checkpointing: true
output_dir: $EXPERIMENT_RESULTS_DIR/sft
do_eval: true
eval_strategy: "no"
log_level: info
seed: $SFT_SEED
SFTEOF
        
        echo "  Submitting SFT validation job..."
        SFT_JOB_ID=$(sbatch \
            --job-name="val_sft_${CONFIG_NAME}" \
            --output="${LOG_DIR}/val_sft_${CONFIG_NAME}_%j.out" \
            --error="${LOG_DIR}/val_sft_${CONFIG_NAME}_%j.err" \
            --parsable \
            /data/cat/ws/hama901h-RL/hpopt/slurm_validate_sft.sh "$SFT_VAL_CONFIG")
        echo "  SFT Job ID: $SFT_JOB_ID"
        SUBMITTED_JOBS+=("$SFT_JOB_ID:sft:$CONFIG_NAME")
    else
        echo "  WARNING: SFT checkpoint not found at $SFT_CHECKPOINT"
    fi
    
    # -------------------------------------------------------------------------
    # Submit DPO Validation Job
    # -------------------------------------------------------------------------
    if [ -d "$DPO_CHECKPOINT" ] && [ -f "$DPO_CHECKPOINT/config.json" ]; then
        if [ ! -d "$SFT_CHECKPOINT" ] || [ ! -f "$SFT_CHECKPOINT/config.json" ]; then
            echo "  WARNING: Reference SFT checkpoint not found at $SFT_CHECKPOINT. Skipping DPO validation."
        else
            DPO_VAL_CONFIG="${TMP_CONFIG_DIR}/dpo_val_${CONFIG_NAME}.yaml"
            cat > "$DPO_VAL_CONFIG" << DPOEOF
# Auto-generated DPO validation config
model_name_or_path: $DPO_CHECKPOINT
ref_model_name_or_path: $SFT_CHECKPOINT
torch_dtype: $TORCH_DTYPE

dataset_name:
  $DPO_DATASET

bf16: true
beta: $DPO_BETA
max_length: $DPO_MAX_LENGTH
max_prompt_length: $DPO_MAX_PROMPT_LENGTH
per_device_eval_batch_size: $DPO_BATCH_SIZE
gradient_checkpointing: true
output_dir: $EXPERIMENT_RESULTS_DIR/dpo
do_eval: true
eval_strategy: "no"
log_level: info
seed: $DPO_SEED
DPOEOF
            
            echo "  Submitting DPO validation job..."
            DPO_JOB_ID=$(sbatch \
                --job-name="val_dpo_${CONFIG_NAME}" \
                --output="${LOG_DIR}/val_dpo_${CONFIG_NAME}_%j.out" \
                --error="${LOG_DIR}/val_dpo_${CONFIG_NAME}_%j.err" \
                --parsable \
                /data/cat/ws/hama901h-RL/hpopt/slurm_validate_dpo.sh "$DPO_VAL_CONFIG")
            echo "  DPO Job ID: $DPO_JOB_ID"
            SUBMITTED_JOBS+=("$DPO_JOB_ID:dpo:$CONFIG_NAME")
        fi
    else
        echo "  WARNING: DPO checkpoint not found at $DPO_CHECKPOINT"
    fi
done

# Save submitted jobs list
JOBS_FILE="${RESULTS_BASE_DIR}/submitted_jobs.txt"
echo "# Submitted validation jobs - $(date)" > "$JOBS_FILE"
for job_info in "${SUBMITTED_JOBS[@]}"; do
    echo "$job_info" >> "$JOBS_FILE"
done

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo "Jobs list saved to: $JOBS_FILE"
echo "Results will be saved to: $RESULTS_BASE_DIR"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To collect results after completion:"
echo "  python3 /data/cat/ws/hama901h-RL/hpopt/collect_validation_results.py"
echo "=============================================="