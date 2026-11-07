#!/bin/bash
#SBATCH --job-name=rlvr-gsm8k
#SBATCH --output=/gpfs/projects/ehpc533/users/hannan/rlvr/logs/%x_%j.out
#SBATCH --error=/gpfs/projects/ehpc533/users/hannan/rlvr/logs/%x_%j.err
#SBATCH --account=ehpc533
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#
# MareNostrum 5 (4x H100-64GB) version of train_grpo.sbatch. Same RL run as on capella: the global
# batch stays 4 GPUs x 4 per device x 16 accumulation = 256 completions = 32 prompts x 8 per
# optimizer step, one generation round per step. Only memory knobs change (64 GB cards): half the
# per-device batch of capella.
# MN5 storage is separate from ZIH, so the .claim guard cannot see capella/alpha copies: cancel those
# for any alias trained here.
#
# Usage: sbatch --job-name=rlvr-<alias> train_grpo_mn5.sbatch <alias> [overrides]
#        (submit_train_mn5.sh submits all five)
set -euo pipefail

ALIAS=$1
R=/gpfs/projects/ehpc533/users/hannan/rlvr
ZIH_PATH=$(awk -F'\t' -v a="$ALIAS" '$2 == a {print $3}' "$R/code/starts.tsv")
[[ -n "$ZIH_PATH" ]] || { echo "alias $ALIAS not in starts.tsv" >&2; exit 1; }
MODEL=$R/dpo-starts/$(basename "$ZIH_PATH")
OUT=$R/RLVR-GSM8K/$ALIAS
echo "START $ALIAS  $MODEL -> $OUT (unless overridden)  $(date)"

source $R/venv/bin/activate
export HF_HOME=$R/hf_cache
export HF_DATASETS_CACHE=$HF_HOME
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$R/cache
export TRITON_CACHE_DIR=$R/cache/triton
export VLLM_CACHE_ROOT=$R/cache/vllm

cd $R/code
accelerate launch --config_file accelerate_fsdp.yaml --num_processes "$SLURM_GPUS_ON_NODE" \
    grpo_gsm8k.py --config grpo_gsm8k.yaml \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
    --model_name_or_path "$MODEL" --output_dir "$OUT" "${@:2}"

echo "END $ALIAS $(date)"
