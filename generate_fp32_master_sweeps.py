#!/usr/bin/env python3
"""Write fp32-master-weight reruns of the small-model Dolci DPO grids (plain DDP).

The original Qwen3-0.6B/1.7B and Llama-3.2-1B/3B runs loaded the model with torch_dtype: bfloat16 under
plain DDP, so AdamW updated bf16 weights with bf16 states and most updates rounded away. The 4B+ runs used
DeepSpeed ZeRO, which keeps fp32 master weights. These reruns load the model in float32 and keep
bf16: true (autocast compute), i.e. fp32 weights + fp32 Adam states, like the ZeRO runs.

Each config is the original run's config with only these changes:
  torch_dtype: float32
  save_only_model: true    (fp32 optimizer states would add ~8 B/param to every checkpoint, ~9 TB in total;
                            they are not needed for evaluation, but a killed job cannot resume)
  model_name_or_path / output_dir: Llama SFT copies on /data/horse; outputs get a "-FP32" suffix
Same data, seed, grid, launcher (ddp.yaml), 2 nodes x 4 GPUs, global batch 128.

Does not submit jobs. Each output directory gets a submit_sweep.sh (supports --dry-run).
--alpha writes single-node alpha versions of the Qwen3-0.6B / 1.7B grids instead (see ALPHA).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "qwen3"))
from generate_ipo_apo_norm_sweeps import ACCEL_ROOT, JOB, write_exec  # noqa: E402  (same capella job template)

LOGS = Path("/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs")
LLAMA = Path("/data/horse/ws/hama901h-Post-training/checkpoints")

MODELS = {
    "qwen3-0.6b": {"src": ROOT / "qwen3/0.6B/DPO_AO", "dst": ROOT / "qwen3/0.6B/DPO_AO_FP32",
                   "log": LOGS / "Qwen3/0.6B/DPO-AO-FP32/SFT-LR3e-5", "name": "Qwen3-0.6B-SFT-LR3e-5-AO"},
    "qwen3-1.7b": {"src": ROOT / "qwen3/1.7B/DPO_AO", "dst": ROOT / "qwen3/1.7B/DPO_AO_FP32",
                   "log": LOGS / "Qwen3/1.7B/DPO-AO-FP32/SFT-LR3e-5", "name": "Qwen3-1.7B-SFT-LR3e-5-AO"},
    "llama3.2-1b": {"src": ROOT / "llama/1b/DPO", "dst": ROOT / "llama/1b/DPO_FP32",
                    "log": LOGS / "Llama/1B/DPO-FP32/SFT-Lr3e-5-AO", "name": "Llama-3.2-1B-SFT-Lr3e-5-AO",
                    "sft": LLAMA / "Llama-3.2-1B/SFT-Lr3e-5-AO"},
    "llama3.2-3b": {"src": ROOT / "llama/3b/DPO", "dst": ROOT / "llama/3b/DPO_FP32",
                    "log": LOGS / "Llama/3B/DPO-FP32/SFT-Lr3e-5-AO", "name": "Llama-3.2-3B-SFT-Lr3e-5-AO",
                    "sft": LLAMA / "Llama-3.2-3B/SFT-Lr3e-5-AO"},
}
CONFIG_RE = re.compile(r"dpo_beta([\d.]+)_LR(\d+e-\d+)\.yaml")   # skips the dpo_beta0.1_LR.yaml template

SUBMIT = r"""#!/bin/bash
set -euo pipefail
# Submit every fp32-master DPO job script in this directory. Does not regenerate configs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
fi
shopt -s nullglob
count=0
for job in "${SCRIPT_DIR}"/DPO-*-FP32.sh; do
  count=$((count + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN: sbatch $job"
  else
    sbatch "$job"
  fi
done
echo "$count job scripts."
"""


def fp32_config(text: str, spec: dict, job_name: str) -> str:
    sft = spec.get("sft") or Path(re.search(r"^model_name_or_path: (.+)$", text, re.M).group(1).strip())
    out = (sft / job_name) if "sft" in spec else (sft / "olmo3-dpo" / job_name)
    text = re.sub(r"^model_name_or_path: .*$", f"model_name_or_path: {sft}", text, flags=re.M)
    text = re.sub(r"^torch_dtype: .*$", "torch_dtype: float32", text, flags=re.M)
    text = re.sub(r"^output_dir: .*$", f"output_dir: {out}/", text, flags=re.M)
    return text.rstrip("\n") + "\nsave_only_model: true\n"


# Alpha (1 node x 8 A100-40GB, grad accum 16 -> global batch 128, as on capella). Each job gets its own output_dir.
# 1.7B does not fit fp32 DDP on 40 GB (~41 GB static: fp32 weights + grads + Adam + DDP buckets + fp32 ref), so it
# runs the original bf16 config under ZeRO-3, which keeps fp32 master weights like the 4B/8B/14B runs.
ALPHA = {"qwen3-0.6b": {"mode": "ddp-fp32", "suffix": "FP32-alpha", "time": "12:00:00"},
         "qwen3-1.7b": {"mode": "zero3", "suffix": "ZeRO3-alpha", "time": "1-00:00:00"}}
ALPHA_HEADER = [("#SBATCH --nodes=2", "#SBATCH --nodes=1"), ("#SBATCH --gres=gpu:4", "#SBATCH --gres=gpu:8"),
                ("#SBATCH --cpus-per-task=14", "#SBATCH --cpus-per-task=16"),
                ("#SBATCH --partition=capella", "#SBATCH --partition=alpha\n#SBATCH --exclusive"),
                ("export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'",
                 "export NCCL_SOCKET_IFNAME='ib0.0002,ib1.0002'\n"
                 # 40 GB is tight for a 4096-token pair (fp32 logits alone 4.6 GiB); job 2382250 OOMed on fragmentation
                 "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")]
SUBMIT_ALPHA = r"""#!/bin/bash
set -euo pipefail
# Submit every alpha job script in this directory, skipping any whose output_dir already exists
# (finished, running or partial), so no run can overwrite another. Does not regenerate configs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
fi
shopt -s nullglob
count=0
for job in "${SCRIPT_DIR}"/DPO-*-alpha.sh; do
  cfg="$(grep -m1 '^CONFIG_FILE=' "$job" | cut -d= -f2)"
  out="$(grep -m1 '^output_dir:' "$cfg" | awk '{print $2}')"
  if [[ -d "$out" ]]; then
    echo "SKIP (output_dir exists): $(basename "$job")"
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


def alpha_job(**kw) -> str:
    text = JOB.format(**kw)
    for old, new in ALPHA_HEADER:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def main_alpha() -> None:
    n = 0
    for key, a in ALPHA.items():
        spec = MODELS[key]
        dst = spec["dst"].parent / "DPO_AO_FP32_alpha"
        dst.mkdir(parents=True, exist_ok=True)
        for src in sorted(spec["src"].glob("dpo_beta*_LR*.yaml")):
            m = CONFIG_RE.fullmatch(src.name)
            if not m:
                continue
            beta, lr = m.groups()
            job_name = f"DPO-LR{lr}_Beta{beta}-{spec['name']}-{a['suffix']}"
            if a["mode"] == "ddp-fp32":
                text, accel = fp32_config(src.read_text(), spec, job_name), ACCEL_ROOT / "ddp.yaml"
            else:   # original bf16 config; ZeRO-3 provides the fp32 master weights
                text = src.read_text()
                sft = Path(re.search(r"^model_name_or_path: (.+)$", text, re.M).group(1).strip())
                text = re.sub(r"^output_dir: .*$", f"output_dir: {sft / 'olmo3-dpo' / job_name}/", text, flags=re.M)
                text, accel = text.rstrip("\n") + "\nsave_only_model: true\n", ACCEL_ROOT / "zero3.yaml"
            config_path = dst / src.name
            config_path.write_text(text)
            write_exec(dst / f"{job_name}.sh", alpha_job(job_name=job_name, log_dir=spec["log"], time=a["time"],
                                                         accel=accel, config_path=config_path))
            n += 1
        write_exec(dst / "submit_sweep.sh", SUBMIT_ALPHA)
        print(f"{key} (alpha, {a['mode']}): {dst}")
    print(f"wrote {n} alpha jobs")


def main() -> None:
    n = 0
    for key, spec in MODELS.items():
        spec["dst"].mkdir(parents=True, exist_ok=True)
        spec["log"].mkdir(parents=True, exist_ok=True)
        for src in sorted(spec["src"].glob("dpo_beta*_LR*.yaml")):
            m = CONFIG_RE.fullmatch(src.name)
            if not m:
                continue
            beta, lr = m.groups()
            job_name = f"DPO-LR{lr}_Beta{beta}-{spec['name']}-FP32"
            config_path = spec["dst"] / src.name
            config_path.write_text(fp32_config(src.read_text(), spec, job_name))
            write_exec(spec["dst"] / f"{job_name}.sh",
                       JOB.format(job_name=job_name, log_dir=spec["log"], time="12:00:00",
                                  accel=ACCEL_ROOT / "ddp.yaml", config_path=config_path))
            n += 1
        write_exec(spec["dst"] / "submit_sweep.sh", SUBMIT)
        print(f"{key}: {spec['dst']}")
    print(f"wrote {n} jobs")


if __name__ == "__main__":
    main_alpha() if "--alpha" in sys.argv else main()
