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
    main()
