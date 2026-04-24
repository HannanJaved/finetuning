#!/usr/bin/env python3
"""Compute average token-level Shannon entropy for SFT checkpoints on a probe set."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    path: Path
    size: Optional[str]
    lr: Optional[str]
    step: Optional[int]


@dataclass
class PromptSource:
    prompts: Optional[List[str]] = None
    dataset: Optional[object] = None
    use_full_dataset: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average token-level Shannon entropy for SFT checkpoints.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help=(
            "Root folder that contains Qwen SFT runs. Example: "
            "/data/horse/ws/.../SFT-sweep/"
        ),
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="Qwen3-*-SFT-LR*",
        help="Glob to find run folders under checkpoint root.",
    )
    parser.add_argument(
        "--checkpoint-subglob",
        default="checkpoint-*",
        help=(
            "Glob inside each run folder to select checkpoints. "
            "Use 'none' to use the run folder itself if it contains a model."
        ),
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only compute entropy for the final checkpoint per run.",
    )
    parser.add_argument(
        "--dataset",
        default="nvidia/Nemotron-Post-Training-Dataset-v2",
        help="Hugging Face dataset name for probe prompts.",
    )
    parser.add_argument(
        "--dataset-config",
        default="SFT",
        help="Dataset config name (e.g., SFT).",
    )
    parser.add_argument(
        "--dataset-split",
        default="code,math",
        help="Comma-separated dataset splits to sample prompts from.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=128,
        help="Number of prompts to sample for the probe set.",
    )
    parser.add_argument(
        "--use-full-dataset",
        action="store_true",
        help="Use the full dataset (no probe sampling).",
    )
    parser.add_argument(
        "--prompt-fraction",
        type=float,
        default=None,
        help="If set, sample this fraction of the dataset for probe prompts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt sampling.",
    )
    parser.add_argument(
        "--probe-cache",
        type=Path,
        default=Path("probe_prompts.jsonl"),
        help="Path to cache/load the fixed probe prompts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for model forward passes.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("entropy_results.csv"),
        help="Where to write the entropy summary CSV.",
    )
    return parser.parse_args()


def resolve_device(device_choice: str) -> torch.device:
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_choice)


def resolve_dtype(dtype_choice: str, device: torch.device) -> torch.dtype:
    if dtype_choice == "auto":
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    return getattr(torch, dtype_choice)


def iter_run_folders(root: Path, run_glob: str) -> Iterable[Path]:
    for path in sorted(root.glob(run_glob)):
        if path.is_dir():
            yield path


def iter_checkpoints(run_dir: Path, subglob: str) -> Iterable[Path]:
    if subglob == "none":
        yield run_dir
        return
    checkpoints = sorted(run_dir.glob(subglob))
    for ckpt in checkpoints:
        if ckpt.is_dir():
            yield ckpt


def parse_run_metadata(path: Path) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(r"Qwen3-(?P<size>[\d\.]+B).*SFT-LR(?P<lr>[\deE\-\.]+)", path.name)
    if not match:
        return None, None
    return match.group("size"), match.group("lr")


def parse_step(path: Path) -> Optional[int]:
    match = re.search(r"checkpoint-(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def load_prompt_source(
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_prompts: int,
    prompt_fraction: Optional[float],
    seed: int,
    cache_path: Path,
    use_full_dataset: bool,
) -> PromptSource:
    splits = [item.strip() for item in split.split(",") if item.strip()]
    if len(splits) == 1:
        dataset = load_dataset(dataset_name, dataset_config, split=splits[0])
    else:
        datasets_list = load_dataset(dataset_name, dataset_config, split=splits)
        dataset = concatenate_datasets(datasets_list)

    if use_full_dataset:
        LOGGER.info(
            "Using full dataset %s/%s splits=%s (no probe sampling)",
            dataset_name,
            dataset_config,
            ",".join(splits),
        )
        return PromptSource(dataset=dataset, use_full_dataset=True)

    if cache_path.exists():
        LOGGER.info("Loading cached probe prompts from %s", cache_path)
        with cache_path.open("r", encoding="utf-8") as handle:
            return PromptSource(prompts=[json.loads(line)["prompt"] for line in handle])

    LOGGER.info(
        "Sampling %s prompts from %s/%s splits=%s",
        num_prompts,
        dataset_name,
        dataset_config,
        ",".join(splits),
    )

    if prompt_fraction is not None:
        if prompt_fraction <= 0 or prompt_fraction > 1:
            raise ValueError("--prompt-fraction must be in (0, 1].")
        target = max(1, int(len(dataset) * prompt_fraction))
    else:
        target = num_prompts

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=min(target, len(dataset)))
    prompts: List[str] = []
    for idx in indices:
        row = dataset[idx]
        prompt = extract_prompt(row)
        if prompt:
            prompts.append(prompt)
    if len(prompts) < target:
        LOGGER.warning("Only collected %s prompts (requested %s)", len(prompts), target)

    with cache_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    return PromptSource(prompts=prompts)


def extract_prompt(row: dict) -> Optional[str]:
    if "prompt" in row:
        return row["prompt"]
    if "instruction" in row:
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        if input_text:
            return f"{instruction}\n\n{input_text}".strip()
        return instruction.strip()
    if "messages" in row and row["messages"]:
        for msg in row["messages"]:
            if msg.get("role") == "user":
                return msg.get("content", "").strip()
        return row["messages"][0].get("content", "").strip()
    if "conversations" in row and row["conversations"]:
        for msg in row["conversations"]:
            if msg.get("from") in {"human", "user"}:
                return msg.get("value", "").strip()
        return row["conversations"][0].get("value", "").strip()
    return None


def chunked(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def iter_prompt_batches(source: PromptSource, batch_size: int) -> Tuple[Iterable[List[str]], Optional[int]]:
    if not source.use_full_dataset:
        prompts = source.prompts or []
        return chunked(prompts, batch_size), len(prompts)

    dataset = source.dataset
    if dataset is None:
        raise ValueError("PromptSource dataset is missing for full-dataset mode.")

    def generator() -> Iterable[List[str]]:
        batch: List[str] = []
        for row in dataset:
            prompt = extract_prompt(row)
            if not prompt:
                continue
            batch.append(prompt)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    return generator(), None


def compute_entropy_for_checkpoint(
    checkpoint: Path,
    prompt_source: PromptSource,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_length: int,
) -> Tuple[float, int, int]:
    LOGGER.info("Loading model from %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    total_entropy = 0.0
    total_tokens = 0

    batches, prompt_count = iter_prompt_batches(prompt_source, batch_size)
    processed_prompts = 0

    with torch.no_grad():
        for batch in tqdm(list(batches) if prompt_count is not None else batches, desc=checkpoint.name):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)

            mask = attention_mask.clone()
            if mask.shape[1] > 0:
                mask[:, 0] = 0
            entropy = entropy * mask
            total_entropy += entropy.sum().item()
            total_tokens += int(mask.sum().item())
        processed_prompts += len(batch)

    avg_entropy = total_entropy / max(total_tokens, 1)
    return avg_entropy, total_tokens, processed_prompts


def build_checkpoint_index(
    root: Path,
    run_glob: str,
    subglob: str,
    latest_only: bool,
) -> List[CheckpointInfo]:
    checkpoints: List[CheckpointInfo] = []
    for run in iter_run_folders(root, run_glob):
        size, lr = parse_run_metadata(run)
        run_checkpoints: List[CheckpointInfo] = []
        for ckpt in iter_checkpoints(run, subglob):
            if not (ckpt / "config.json").exists():
                continue
            step = parse_step(ckpt)
            run_checkpoints.append(CheckpointInfo(path=ckpt, size=size, lr=lr, step=step))

        if not run_checkpoints:
            continue

        if latest_only:
            with_steps = [ck for ck in run_checkpoints if ck.step is not None]
            if with_steps:
                chosen = max(with_steps, key=lambda ck: ck.step)
            else:
                chosen = sorted(run_checkpoints, key=lambda ck: ck.path.name)[-1]
            checkpoints.append(chosen)
        else:
            checkpoints.extend(run_checkpoints)
    return checkpoints


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    prompt_source = load_prompt_source(
        args.dataset,
        args.dataset_config,
        args.dataset_split,
        args.num_prompts,
        args.prompt_fraction,
        args.seed,
        args.probe_cache,
        args.use_full_dataset,
    )

    checkpoints = build_checkpoint_index(
        args.checkpoint_root,
        args.checkpoint_glob,
        args.checkpoint_subglob,
        args.latest_only,
    )
    if not checkpoints:
        raise RuntimeError("No checkpoints found. Check --checkpoint-root and glob settings.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8") as handle:
        handle.write(
            "size,lr,checkpoint,checkpoint_step,avg_entropy,num_tokens,num_prompts,dataset,dataset_split\n"
        )
        for ckpt in checkpoints:
            avg_entropy, num_tokens, num_prompts = compute_entropy_for_checkpoint(
                ckpt.path,
                prompt_source,
                device,
                dtype,
                args.batch_size,
                args.max_length,
            )
            handle.write(
                f"{ckpt.size},{ckpt.lr},{ckpt.path},"
                f"{ckpt.step if ckpt.step is not None else ''},"
                f"{avg_entropy:.6f},{num_tokens},{num_prompts},"
                f"{args.dataset},{args.dataset_split}\n"
            )
            handle.flush()

    LOGGER.info("Wrote results to %s", args.output_csv)


if __name__ == "__main__":
    main()
