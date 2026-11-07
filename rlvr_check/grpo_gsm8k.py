#!/usr/bin/env python
"""RLVR stage: GRPO on GSM8K train with an exact-match reward (no reward model).

All hyperparameters live in grpo_gsm8k.yaml so every starting point is trained identically;
only --model_name_or_path and --output_dir differ (set by train_grpo.sbatch).
"""
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, TrlParser

INSTRUCTION = "Solve the problem step by step. End your response with the final answer in the form '#### <number>'."
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class ScriptArgs:
    model_name_or_path: str = field(metadata={"help": "Starting checkpoint (DPO or SFT)."})


def last_number(text):
    numbers = NUMBER.findall(text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def exact_match_reward(completions, answer, **kwargs):
    """1.0 iff the last number in the completion equals the gold answer (Tulu 3 / open-instruct GSM8K verifier)."""
    return [float(last_number(c[0]["content"]) == float(gold)) for c, gold in zip(completions, answer)]


def claim_or_exit(output_dir):
    """Only the first Slurm job to start trains into output_dir; later duplicates (other account or
    cluster) exit 0. The claim is an atomic hard link on the shared filesystem, so it works across
    capella and alpha. Delete <output_dir>.claim to allow a rerun after a failed run."""
    job = os.environ.get("SLURM_JOB_ID")
    if job is None:
        return
    me = f"{os.environ.get('SLURM_CLUSTER_NAME', '?')}:{job}"
    claim = Path(output_dir.rstrip("/") + ".claim")
    claim.parent.mkdir(parents=True, exist_ok=True)
    tmp = claim.with_name(f"{claim.name}.{me.replace(':', '_')}.{os.getpid()}")
    tmp.write_text(me)
    try:
        os.link(tmp, claim)
    except FileExistsError:
        pass
    finally:
        tmp.unlink()
    owner = claim.read_text()
    if owner != me:
        print(f"{output_dir} is claimed by job {owner}; this duplicate exits.", flush=True)
        sys.exit(0)


def to_prompt(example):
    return {
        "prompt": [{"role": "user", "content": f"{example['question']}\n\n{INSTRUCTION}"}],
        "answer": example["answer"].split("####")[-1].strip().replace(",", ""),
    }


def main():
    script_args, cfg = TrlParser((ScriptArgs, GRPOConfig)).parse_args_and_config()
    claim_or_exit(cfg.output_dir)
    train =load_dataset("openai/gsm8k", "main", split="train")
    train = train.map(to_prompt, remove_columns=train.column_names)

    trainer = GRPOTrainer(model=script_args.model_name_or_path, reward_funcs=exact_match_reward,
                          args=cfg, train_dataset=train)
    trainer.train()

    # Training keeps fp32 master weights; export bf16 like the SFT/DPO checkpoints it is compared with.
    fp32_dir = cfg.output_dir.rstrip("/") + "-fp32"
    trainer.save_model(fp32_dir)
    if trainer.accelerator.is_main_process:
        AutoModelForCausalLM.from_pretrained(fp32_dir, dtype=torch.bfloat16).save_pretrained(cfg.output_dir)
        AutoTokenizer.from_pretrained(fp32_dir).save_pretrained(cfg.output_dir)
        shutil.rmtree(fp32_dir)


if __name__ == "__main__":
    main()
