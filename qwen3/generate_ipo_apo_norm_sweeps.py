#!/usr/bin/env python3
"""Write IPO, APO, and length-normalized DPO sweeps for Qwen3 1.7B, 4B, and 8B.

Same data, init, and LR/beta grid as the Dolci DPO_AO sweep:
  dataset: allenai/Dolci-Instruct-DPO
  init:    Qwen3-{size}-SFT-LR3e-5 Assistant-Only
  lrs:     1e-6, 2e-6, 4e-6
  betas:   0.01, 0.02, 0.04

Does not submit jobs. Each variant directory gets a submit_sweep.sh.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_ROOT = Path("/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3")
ACCEL_ROOT = Path(
    "/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/recipes/accelerate_configs"
)
CKPT = Path("/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen")

LRS = ("1e-6", "2e-6", "4e-6")
BETAS = ("0.01", "0.02", "0.04")

SIZES = {
    "1.7B": {
        "model": CKPT / "Qwen3-1.7B-Base/SFT-sweep/Qwen3-1.7B-SFT-LR3e-5-Assistant-Only",
        "accel": "ddp.yaml",
        "time": "12:00:00",
    },
    "4B": {
        "model": CKPT / "Qwen3-4B-Base/SFT-sweep/Qwen3-4B-SFT-LR3e-5/Assistant-Only",
        "accel": "zero3.yaml",
        "time": "12:00:00",
    },
    "8B": {
        "model": CKPT / "Qwen3-8B-Base/SFT-sweep/Qwen3-8B-SFT-LR3e-5/Assistant-Only",
        "accel": "zero3.yaml",
        "time": "1-00:00:00",
    },
}

# APO-zero: chosen responses are better than the SFT policy's own outputs.
# APO-down: even the chosen responses are worse than the SFT policy's outputs.
VARIANTS = (
    {
        "dir": "DPO_IPO",
        "log": "DPO-IPO",
        "tag": "IPO",
        "loss_type": "ipo",
        "length_normalize": False,
        "note": "TRL loss_type=ipo. beta is IPO's tau.",
    },
    {
        "dir": "DPO_APO",
        "log": "DPO-APO",
        "tag": "APOzero",
        "loss_type": "apo_zero",
        "length_normalize": False,
        "note": "TRL loss_type=apo_zero. Chosen is better than the SFT policy.",
    },
    {
        "dir": "DPO_APO",
        "log": "DPO-APO",
        "tag": "APOdown",
        "loss_type": "apo_down",
        "length_normalize": False,
        "note": "TRL loss_type=apo_down. Chosen is still worse than the SFT policy.",
    },
    {
        "dir": "DPO_NORM",
        "log": "DPO-NORM",
        "tag": "NormDPO",
        "loss_type": "sigmoid",
        "length_normalize": True,
        "note": "Sigmoid DPO with length-normalized log-probs (NormDPOTrainer).",
    },
)

JOB = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time={time}
#SBATCH --partition=capella

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA
source /data/horse/ws/hama901h-BFTranslation/venv-post-training/bin/activate

export HF_HOME="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
export HF_DATASETS_CACHE="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.cache"
source /data/horse/ws/hama901h-Post-training/cache.sh
export PYTHONPATH="/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/src:/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook:/data/horse/ws/hama901h-BFTranslation/venv-post-training/lib/python3.11/site-packages"

# Get master node hostname for distributed training
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
export TORCHELASTIC_MAX_FAILED_CONNECTIONS=60
export TORCH_DISTRIBUTED_HEARTBEAT_TIMEOUT=300
export TORCH_DISTRIBUTED_COODINATOR_TIMEOUT=300
export OMP_NUM_THREADS=18

#Distributed variables
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${{nodes_array[0]}}

export RDZV_HOST=$head_node
export RDZV_PORT=29400

echo "head_node=$head_node"

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

echo NPROC_PER_NODE=$NPROC_PER_NODE

# Wandb settings
export WANDB_PROJECT=instruction-tuning
export WANDB_ENTITY=openeurollm-project
export WANDB_NAME={job_name}

cd /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/
ACCELERATE_CONFIG_FILE={accel}
CONFIG_FILE={config_path}

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

#LAUNCHERS
export CMD="scripts/dpo.py --config $CONFIG_FILE"

SRUN_ARGS=" \\
    --wait=60 \\
    --kill-on-bad-exit=1 \\
    "

export ACC_LAUNCHER="accelerate launch \\
    --rdzv_conf \\"rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\\" \\
    --config_file $ACCELERATE_CONFIG_FILE \\
    --num_machines $SLURM_NNODES \\
    --num_processes $WORLD_SIZE \\
    --main_process_ip $MASTER_ADDR \\
    --main_process_port $MASTER_PORT \\
    --machine_rank \\$SLURM_PROCID \\
    --role \\$(hostname -s|tr -dc '0-9'): \\
    --tee 3 \\
    "


srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \\$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
"""

SUBMIT = r"""#!/bin/bash
set -euo pipefail
# Submit job scripts in this directory. Skips APO-down. Does not regenerate configs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
fi
shopt -s nullglob
jobs=("${SCRIPT_DIR}"/*.sh)
count=0
for job in "${jobs[@]}"; do
  base="$(basename "$job")"
  if [[ "$base" == "submit_sweep.sh" || "$base" == APOdown-* ]]; then
    continue
  fi
  count=$((count + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN: sbatch $job"
  else
    sbatch "$job"
  fi
done
echo "$count job scripts."
"""


def yaml_for(variant: dict, model: Path, job_name: str, lr: str, beta: str) -> str:
    output = model / "olmo3-dpo" / job_name
    length_normalize = "true" if variant["length_normalize"] else "false"
    return f"""# {variant["note"]}
model_name_or_path: {model}
torch_dtype: bfloat16
use_liger_kernel: false
attn_implementation: flash_attention_2
dataset_name: allenai/Dolci-Instruct-DPO
bf16: true
beta: {beta}
loss_type:
  - {variant["loss_type"]}
do_eval: false
gradient_checkpointing_kwargs:
  use_reentrant: False
learning_rate: {lr}
log_level: info
logging_steps: 10
lr_scheduler_type: linear
max_length: 4096
max_prompt_length: 512
num_train_epochs: 1
optim: adamw_torch
output_dir: {output}/
overwrite_output_dir: true
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
save_strategy: "steps"
save_steps: 200
save_total_limit: 20
seed: 8
warmup_ratio: 0.1
length_normalize_logps: {length_normalize}
"""


def write_exec(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> None:
    n = 0
    touched_dirs: set[Path] = set()
    for size, spec in SIZES.items():
        model = Path(spec["model"])
        accel = ACCEL_ROOT / spec["accel"]
        for variant in VARIANTS:
            out_dir = ROOT / size / variant["dir"]
            log_dir = LOG_ROOT / size / variant["log"] / "SFT-LR3e-5"
            out_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            touched_dirs.add(out_dir)
            for lr in LRS:
                for beta in BETAS:
                    job_name = f"{variant['tag']}-LR{lr}_Beta{beta}-Qwen3-{size}-SFT-LR3e-5-AO"
                    config_name = f"{variant['tag'].lower()}_beta{beta}_LR{lr}.yaml"
                    config_path = out_dir / config_name
                    config_path.write_text(yaml_for(variant, model, job_name, lr, beta))
                    write_exec(
                        out_dir / f"{job_name}.sh",
                        JOB.format(
                            job_name=job_name,
                            log_dir=log_dir,
                            time=spec["time"],
                            accel=accel,
                            config_path=config_path,
                        ),
                    )
                    n += 1
    for out_dir in sorted(touched_dirs):
        write_exec(out_dir / "submit_sweep.sh", SUBMIT)
    print(f"wrote {n} jobs in {len(touched_dirs)} directories")


if __name__ == "__main__":
    main()
