#!/bin/bash
#SBATCH --job-name=dpo_8b_tulu-pref-mix
#SBATCH --output=/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-finetuning/.logs/TRL/%x_%j.out
#SBATCH --error=/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-finetuning/.logs/TRL/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --partition dev_accelerated-h100
#SBATCH -A hk-project-p0024043

echo "JOB NAME" $SLURM_JOB_NAME

source /home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-finetuning/handbook/bin/activate

export HF_HOME="/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-finetuning/.cache/huggingface"
export HF_DATASETS_CACHE="/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-finetuning/.cache/huggingface/datasets"
export PYTHONPATH="/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-finetuning/handbook/lib/python3.11/site-packages"
#pip show transformers

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

# Wandb settings
# export WANDB_PROJECT=instruction-tuning
# export WANDB_ENTITY=openeurollm-project-

cd /home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-finetuning/alignment-handbook
ACCELERATE_CONFIG_FILE=recipes/accelerate_configs/zero3.yaml
CONFIG_FILE=recipes/llama-3/dpo/config_tulu3_pref_mix_8b.yaml

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

# export CMD="scripts/dpo.py $CONFIG_FILE"
export CMD=" \
    scripts/dpo.py \
    --dataset_name allenai/llama-3.1-tulu-3-8b-preference-mixture \
    --dataset_split train \
    --model_name_or_path /home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-finetuning/checkpoints/openeurollm/datamix-9b-60-40  \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir test_data/dpo \
    --no_remove_unused_columns \
    --bf16 True \
    --bf16_full_eval True
    "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes 4 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    $CMD
# echo "num_machines: $SLURM_NNODES"
# echo "num_processes: $NPROC_PER_NODE"

    
# export ACC_LAUNCHER="accelerate launch \
#     --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
#     --config_file $ACCELERATE_CONFIG_FILE \
#     --num_machines $SLURM_NNODES \
#     --num_processes $NPROC_PER_NODE \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --machine_rank \$SLURM_PROCID \
#     --role \$(hostname -s|tr -dc '0-9'): \
#     --tee 3 \
#     "


# SRUN_ARGS=" \
#     --wait=60 \
#     --kill-on-bad-exit=1 \
#     "

# srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
