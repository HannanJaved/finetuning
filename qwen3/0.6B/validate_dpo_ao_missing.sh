#!/bin/bash
#SBATCH --job-name=Qwen3-0.6B-DPO-VAL-MISSING
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/0.6B/DPO-AO/SFT-LR3e-5/%x_%A_%a.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3/0.6B/DPO-AO/SFT-LR3e-5/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=capella
#SBATCH --array=0-0

set -euo pipefail

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-post-training/bin/activate

export HF_HOME="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
source /data/horse/ws/hama901h-Post-training/cache.sh
export PYTHONPATH="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/src:/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook:/data/horse/ws/hama901h-BFTranslation/venv-post-training/lib/python3.11/site-packages"

export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'
export NCCL_IB_PKEY=0x2
export OMP_NUM_THREADS=18

export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
export RDZV_HOST=$head_node
export RDZV_PORT=29400
echo "head_node=$head_node"
NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
echo NPROC_PER_NODE=$NPROC_PER_NODE

export WANDB_PROJECT=instruction-tuning
export WANDB_ENTITY=openeurollm-project

# Missing cells only (1 jobs)
LRS=(8e-6)
BETAS=(0.01)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

export WANDB_NAME="Qwen3-0.6B-DPO-VAL-LR${LR}-Beta${BETA}"
DPO_DIR="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/qwen3/0.6B/DPO_AO"
CONFIG_FILE="${DPO_DIR}/dpo_beta${BETA}_LR${LR}.yaml"

TRAIN_OUTPUT_DIR=$(grep -E '^output_dir:' "$CONFIG_FILE" | awk '{print $2}')
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR%/}"
REF_MODEL_PATH=$(grep -E '^model_name_or_path:' "$CONFIG_FILE" | awk '{print $2}')
REF_MODEL_PATH="${REF_MODEL_PATH%/}"
LATEST_CHECKPOINT=$(ls -d "${TRAIN_OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
  MODEL_PATH="${LATEST_CHECKPOINT}"
else
  MODEL_PATH="${TRAIN_OUTPUT_DIR}"
fi
VALIDATION_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}/validation"

echo "JOBNAME" $SLURM_JOB_NAME
echo "LR" $LR
echo "BETA" $BETA
echo "CONFIG" $CONFIG_FILE
echo "MODEL" $MODEL_PATH
echo "REF" $REF_MODEL_PATH
echo "VALIDATION_OUTPUT" $VALIDATION_OUTPUT_DIR

cd /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/
ACCELERATE_CONFIG_FILE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/recipes/accelerate_configs/ddp.yaml

export CMD="scripts/validate_dpo.py --config $CONFIG_FILE --model_name_or_path $MODEL_PATH --ref_model_name_or_path $REF_MODEL_PATH --output_dir $VALIDATION_OUTPUT_DIR"

SRUN_ARGS=" --wait=60 --kill-on-bad-exit=1 "

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
