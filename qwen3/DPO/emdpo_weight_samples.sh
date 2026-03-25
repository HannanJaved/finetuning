#!/bin/bash
#SBATCH --job-name=emdpo_weight_samples
#SBATCH --output=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=capella

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

export HF_HOME="/data/cat/ws/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/cat/ws/hama901h-Posttraining/.cache"
export PYTHONPATH="/data/horse/ws/hama901h-BFTranslation/venv-TRL/lib/python3.11/site-packages"

export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export WORLD_SIZE=1

cd /data/cat/ws/hama901h-Posttraining/finetuning/alignment-handbook/
ACCELERATE_CONFIG_FILE=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/plain_1gpu.yaml
CONFIG_FILE=${CONFIG_FILE:-/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/config_emdpo_LR5e7_Beta1_ref2.0_pol1.0_ag0.75_len0.25_eps0.20.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-/data/cat/ws/hama901h-Posttraining/finetuning/alignment-handbook/scripts/emdpo_weight_samples_out}

mkdir -p "$OUTPUT_DIR"

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
echo "OUTPUT" $OUTPUT_DIR
pwd -P

DATASET_FRACTION=${DATASET_FRACTION:-0.1}
export CMD="scripts/emdpo_weight_samples.py --config $CONFIG_FILE --sample_output_dir $OUTPUT_DIR --sample_size 10 --dataset_fraction $DATASET_FRACTION"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

export ACC_LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines 1 \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --tee 3 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"
echo "END $SLURM_JOBID: $(date)"
