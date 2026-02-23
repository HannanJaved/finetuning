#!/bin/bash
#SBATCH --job-name=olmo3-Lr5e4-datamix90-40-9B
#SBATCH --output=/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/.logs/SFT_HP_GRID/%x_%j.out
#SBATCH --error=/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/.logs/SFT_HP_GRID/%x_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --partition accelerated-h100
#SBATCH -A hk-project-p0024043

echo "JOB NAME" $SLURM_JOB_NAME

source /home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/venv-finetuning/bin/activate

export HF_HOME="/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/.cache/huggingface"
export HF_DATASETS_CACHE="/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/.cache/huggingface/datasets"
export PYTHONPATH="/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/venv-finetuning/lib/python3.11/site-packages"
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
export WANDB_PROJECT=instruction-tuning
export WANDB_ENTITY=openeurollm-project

cd /home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/finetuning/alignment-handbook/
ACCELERATE_CONFIG_FILE=/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/finetuning/datamix-90-40-9B/zero3.yaml
CONFIG_FILE=/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/finetuning/datamix-90-40-9B/config_olmo3_sft_Lr5e4.yaml

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

echo "num_machines: $SLURM_NNODES"
echo "num_processes: $NPROC_PER_NODE"


#LAUNCHERS
export CMD="scripts/sft.py --config $CONFIG_FILE"

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

