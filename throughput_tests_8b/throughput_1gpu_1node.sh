#!/bin/bash
#SBATCH --job-name=sft8b_tp_1g1n
#SBATCH --output=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3-8B/throughput/8B/%x_%j.out
#SBATCH --error=/data/cat/ws/hama901h-Posttraining/.logs/Qwen3-8B/throughput/8B/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
GPUS_PER_NODE=1
export WORLD_SIZE=$((GPUS_PER_NODE*SLURM_NNODES))

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

export RDZV_HOST=$head_node
export RDZV_PORT=29400

echo "head_node=$head_node"

cd /data/cat/ws/hama901h-Posttraining/finetuning/alignment-handbook/
ACCELERATE_CONFIG_FILE=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/zero3.yaml
CONFIG_FILE=/data/cat/ws/hama901h-Posttraining/finetuning/throughput_tests_8b/config_throughput.yaml

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

export CMD="scripts/sft.py --config $CONFIG_FILE --output_dir /data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-8B-Base/test_throughput/4gpu_1node/"

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
