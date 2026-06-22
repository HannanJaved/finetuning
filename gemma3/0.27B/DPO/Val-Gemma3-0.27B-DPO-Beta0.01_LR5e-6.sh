#!/bin/bash
#SBATCH --job-name=Val-Gemma3-0.27B-DPO-Beta0.01_LR5e-6
#SBATCH --output=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Gemma3/0.27B/DPO/validate/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Gemma3/0.27B/DPO/validate/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=capella

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-post-training/bin/activate

export HF_HOME="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
source /data/horse/ws/hama901h-Post-training/cache.sh
export PYTHONPATH="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/src:/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook:/data/horse/ws/hama901h-BFTranslation/venv-post-training/lib/python3.11/site-packages"

export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'
export NCCL_IB_PKEY=0x2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_DEBUG=INFO
export NCCL_IB_RETRY_CNT=10
export NCCL_MIN_NCHANNELS=11
export NCCL_TREE_THRESHOLD=4294967296
export OMP_NUM_THREADS=18

export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$(nvidia-smi -L | wc -l)

export WANDB_DISABLED=true

cd /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook
CONFIG_FILE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/gemma3/0.27B/DPO/validate_dpo_beta0.01_LR5e-6_0.27B.yaml

echo "CONFIG" $CONFIG_FILE

srun --wait=60 --kill-on-bad-exit=1 --jobid $SLURM_JOB_ID bash -c "accelerate launch \
    --config_file /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/recipes/accelerate_configs/ddp.yaml \
    --num_machines 1 \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    scripts/validate_dpo.py --config $CONFIG_FILE"

echo "END TIME: $(date)"
