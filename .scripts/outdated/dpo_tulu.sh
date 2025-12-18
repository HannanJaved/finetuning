#!/bin/bash
#SBATCH --job-name=full_DPO_on_Tulu_SFT
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/Tulu/Finetuning/SFT_DPO/%x.out
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/Tulu/Finetuning/SFT_DPO/%x.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/

# Get master node hostname for distributed training
export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'
# try limited membership instead of full
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
export TORCHELASTIC_MAX_FAILED_CONNECTIONS=60
export TORCH_DISTRIBUTED_HEARTBEAT_TIMEOUT=300
export TORCH_DISTRIBUTED_COODINATOR_TIMEOUT=300
export OMP_NUM_THREADS=18

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

export RDZV_HOST=$head_node
export RDZV_PORT=29400

echo "head_node=$head_node"

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

echo NPROC_PER_NODE=$NPROC_PER_NODE


srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    /data/horse/ws/hama901h-BFTranslation/open-instruct/open_instruct/dpo_tune_cache.py \
        --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
        --dataset_mixer_list_splits train \
        --max_seq_length 2048 \
        --model_name_or_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-Tulu-3-8B-SFT \
        --tokenizer_name /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-Tulu-3-8B-SFT \
        --output_dir /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/full_run/DPO_on_Tulu_SFT_2Nodes/ \
        --use_slow_tokenizer \
        --preprocessing_num_workers 16 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-7 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --weight_decay 0.0 \
        --num_train_epochs 1 \
        --dpo_loss_type dpo_norm \
        --dpo_beta 5 \
        --use_flash_attn True \
        --gradient_checkpointing True \
        --with_tracking \
        --logging_steps 1 \
        --validation_split_percentage 1 \
        --validation_steps 10 \
        --seed 8 \
        --report_to wandb \
        --with_tracking \
        --wandb_project_name instruction-tuning \
        --wandb_entity openeurollm-project \
        --exp_name "Tulu3_SFT_Checkpoint_Full_DPO_2_Nodes" \
        --add_seed_and_date_to_exp_name False \
        --checkpointing_steps 100 \
        --dataset_skip_cache True \
        # --use_lora True \
        # --lora_rank 256 \
        # --lora_alpha 64