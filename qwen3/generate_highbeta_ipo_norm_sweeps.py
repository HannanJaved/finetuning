#!/usr/bin/env python3
"""IPO and NormDPO sweeps with betas matched to per-token log-ratios.

IPO beta is tau. The loss targets 1/(2*beta) nats per token:
  0.25 -> 2,  0.5 -> 1,  1.0 -> 0.5
NormDPO divides sequence log-probs by length, then uses sigmoid DPO.
Betas 5, 10, 20 match the old 0.01, 0.02, 0.04 grid on a ~400-token sum.

Learning rates stay 1e-6, 2e-6, 4e-6. Sizes 1.7B, 4B, 8B.
Writes DPO_IPO_HIBETA and DPO_NORM_HIBETA. Does not submit.
"""

from __future__ import annotations

import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_ROOT = Path("/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/Qwen3")
ACCEL_ROOT = Path(
    "/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/recipes/accelerate_configs"
)
CKPT = Path("/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen")

LRS = ("1e-6", "2e-6", "4e-6")

SIZES = {
    "1.7B": {
        "model": CKPT / "Qwen3-1.7B-Base/SFT-sweep/Qwen3-1.7B-SFT-LR3e-5-Assistant-Only",
        "accel": "ddp.yaml",
        "time": "12:00:00",
    },
    "4B": {
        "model": CKPT / "Qwen3-4B-Base/SFT-sweep/Qwen3-4B-SFT-LR3e-5/Assistant-Only",
        "accel": "zero3.yaml",
        "time": "18:00:00",
    },
    "8B": {
        "model": CKPT / "Qwen3-8B-Base/SFT-sweep/Qwen3-8B-SFT-LR3e-5/Assistant-Only",
        "accel": "zero3.yaml",
        "time": "1-00:00:00",
    },
}

VARIANTS = (
    {
        "dir": "DPO_IPO_HIBETA",
        "log": "DPO-IPO-HIBETA",
        "tag": "IPO",
        "loss_type": "ipo",
        "length_normalize": False,
        "betas": ("0.25", "0.5", "1.0"),
        "note": "IPO on per-token log-ratios. beta 0.25/0.5/1.0 targets 2/1/0.5 nats per token.",
    },
    {
        "dir": "DPO_NORM_HIBETA",
        "log": "DPO-NORM-HIBETA",
        "tag": "NormDPO",
        "loss_type": "sigmoid",
        "length_normalize": True,
        "betas": ("5", "10", "20"),
        "note": "Length-normalized sigmoid DPO. beta 5/10/20 matches old 0.01/0.02/0.04 on a ~400-token sum.",
    },
)

JOB = """#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=__LOG_DIR__/%x_%j.out
#SBATCH --error=__LOG_DIR__/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=__TIME__
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
head_node=${nodes_array[0]}

export RDZV_HOST=$head_node
export RDZV_PORT=29400

echo "head_node=$head_node"

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

echo NPROC_PER_NODE=$NPROC_PER_NODE

# Wandb settings
export WANDB_PROJECT=instruction-tuning
export WANDB_ENTITY=openeurollm-project
export WANDB_NAME=__JOB_NAME__

cd /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/alignment-handbook/
ACCELERATE_CONFIG_FILE=__ACCEL__
CONFIG_FILE=__CONFIG__

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

SUBMIT = """#!/bin/bash
set -euo pipefail
# Submit every job script in this directory.
# Usage: ./submit_sweep.sh [account]
# Example: ./submit_sweep.sh p_neurasearch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${1:-}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
  ACCOUNT=""
elif [[ "${2:-}" == "--dry-run" || "${2:-}" == "-n" ]]; then
  DRY_RUN=1
fi
shopt -s nullglob
count=0
for job in "${SCRIPT_DIR}"/*.sh; do
  base="$(basename "$job")"
  if [[ "$base" == "submit_sweep.sh" ]]; then
    continue
  fi
  count=$((count + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ -n "$ACCOUNT" ]]; then
      echo "DRY RUN: sbatch --account=$ACCOUNT $job"
    else
      echo "DRY RUN: sbatch $job"
    fi
  elif [[ -n "$ACCOUNT" ]]; then
    sbatch --account="$ACCOUNT" "$job"
  else
    sbatch "$job"
  fi
done
echo "$count job scripts."
"""

TOP = """#!/bin/bash
set -euo pipefail
# Submit the high-beta IPO and NormDPO sweeps for 1.7B, 4B, and 8B.
# Usage: ./submit_highbeta_ipo_norm.sh [account] [--dry-run]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=("$@")
for size in 1.7B 4B 8B; do
  for variant in DPO_IPO_HIBETA DPO_NORM_HIBETA; do
    echo "== $size $variant =="
    "$ROOT/$size/$variant/submit_sweep.sh" "${args[@]}"
  done
done
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


def render_job(job_name: str, log_dir: Path, time: str, accel: Path, config_path: Path) -> str:
    return (
        JOB.replace("__JOB_NAME__", job_name)
        .replace("__LOG_DIR__", str(log_dir))
        .replace("__TIME__", time)
        .replace("__ACCEL__", str(accel))
        .replace("__CONFIG__", str(config_path))
    )


def main() -> None:
    n = 0
    for size, spec in SIZES.items():
        model = Path(spec["model"])
        accel = ACCEL_ROOT / spec["accel"]
        for variant in VARIANTS:
            out_dir = ROOT / size / variant["dir"]
            log_dir = LOG_ROOT / size / variant["log"] / "SFT-LR3e-5"
            out_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            for lr in LRS:
                for beta in variant["betas"]:
                    job_name = f"{variant['tag']}-LR{lr}_Beta{beta}-Qwen3-{size}-SFT-LR3e-5-AO"
                    config_path = out_dir / f"{variant['tag'].lower()}_beta{beta}_LR{lr}.yaml"
                    config_path.write_text(yaml_for(variant, model, job_name, lr, beta))
                    write_exec(
                        out_dir / f"{job_name}.sh",
                        render_job(job_name, log_dir, spec["time"], accel, config_path),
                    )
                    n += 1
            write_exec(out_dir / "submit_sweep.sh", SUBMIT)
    write_exec(ROOT / "submit_highbeta_ipo_norm.sh", TOP)
    print(f"wrote {n} jobs")


if __name__ == "__main__":
    main()
