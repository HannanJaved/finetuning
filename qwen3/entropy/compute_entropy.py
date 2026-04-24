#!/usr/bin/env python3
"""Compute average token-level Shannon entropy for SFT checkpoints on a probe set.

Entropy is measured over *generated* token positions only, not prompt positions.
This captures the model's output distribution peakiness as intended by the
squeezing-effect analysis: we want to know how peaked the model is when it is
freely generating, not how uncertain it is about the next prompt token.

Procedure per prompt:
  1. Tokenise the prompt.
  2. Greedily generate `--gen-tokens` tokens.
  3. Run a single forward pass on [prompt + generated] to get logits.
  4. Compute per-position entropy from the logits at the generated positions.
  5. Average over all generated token positions across the probe set.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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


@dataclass
class VllmConfig:
    enabled: bool = False
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256


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
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum number of tokens to keep from each prompt.",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=256,
        help=(
            "Number of tokens to generate per prompt. "
            "Entropy is computed over exactly these positions. "
            "Larger values give a more stable estimate but cost more compute."
        ),
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
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help=(
            "Use vLLM to generate continuations (faster). Entropy is still "
            "computed with a HF forward pass on the generated sequences."
        ),
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=4096,
        help="Max model length for vLLM engine (prompt + generation).",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM is allowed to use.",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=256,
        help="Max concurrent sequences for vLLM scheduling.",
    )
    return parser.parse_args()


def resolve_device(device_choice: str) -> torch.device:
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_choice)


def resolve_dtype(dtype_choice: str, device: torch.device) -> torch.dtype:
    if dtype_choice == "auto":
        if device.type == "cuda":
            # bfloat16 matches Qwen3 release dtype; float16 can introduce
            # numerical error on models trained in bf16.
            return torch.bfloat16
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


def iter_prompt_batches(
    source: PromptSource, batch_size: int
) -> Tuple[Iterable[List[str]], Optional[int]]:
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


def build_vllm_config(args: argparse.Namespace) -> Optional[VllmConfig]:
    if not args.use_vllm:
        return None
    return VllmConfig(
        enabled=True,
        max_model_len=args.vllm_max_model_len,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_num_seqs=args.vllm_max_num_seqs,
    )


def tokenize_prompts(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_prompt_length: int,
) -> List[List[int]]:
    encoded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_prompt_length,
        return_attention_mask=False,
    )
    return encoded["input_ids"]


def generate_with_vllm(
    checkpoint: Path,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_prompt_length: int,
    gen_tokens: int,
    vllm_config: VllmConfig,
) -> Tuple[List[List[int]], List[int]]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in the current environment. "
            "Install it or disable --use-vllm."
        ) from exc

    try:
        from vllm.inputs import TokensPrompt
        use_token_prompts = True
    except Exception:
        TokensPrompt = None
        use_token_prompts = False

    LOGGER.info("Generating with vLLM from %s", checkpoint)
    prompt_token_ids = tokenize_prompts(tokenizer, prompts, max_prompt_length)
    prompt_lens = [len(ids) for ids in prompt_token_ids]

    if use_token_prompts and TokensPrompt is not None:
        vllm_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
    else:
        LOGGER.warning(
            "vLLM token prompt API unavailable; falling back to text prompts. "
            "Prompt truncation is applied via tokenizer decoding."
        )
        vllm_prompts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in prompt_token_ids]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=gen_tokens,
    )

    llm = LLM(
        model=str(checkpoint),
        tensor_parallel_size=vllm_config.tensor_parallel_size,
        gpu_memory_utilization=vllm_config.gpu_memory_utilization,
        max_model_len=vllm_config.max_model_len,
        max_num_seqs=vllm_config.max_num_seqs,
        trust_remote_code=True,
    )

    outputs = llm.generate(vllm_prompts, sampling_params=sampling_params)

    full_sequences: List[List[int]] = []
    for prompt_ids, output in zip(prompt_token_ids, outputs):
        if not output.outputs:
            gen_ids = []
        else:
            gen_ids = output.outputs[0].token_ids
        full_sequences.append(prompt_ids + gen_ids)

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return full_sequences, prompt_lens


def batch_full_sequences(
    sequences: List[List[int]],
    prompt_lens: List[int],
    pad_token_id: int,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
    for start in range(0, len(sequences), batch_size):
        batch_ids = sequences[start : start + batch_size]
        batch_prompt_lens = prompt_lens[start : start + batch_size]
        max_len = max(len(ids) for ids in batch_ids)
        input_ids = torch.full(
            (len(batch_ids), max_len), pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros(
            (len(batch_ids), max_len), dtype=torch.long
        )
        for i, ids in enumerate(batch_ids):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
        yield input_ids, attention_mask, batch_prompt_lens


def compute_entropy_for_checkpoint(
    checkpoint: Path,
    prompt_source: PromptSource,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_prompt_length: int,
    gen_tokens: int,
    vllm_config: Optional[VllmConfig] = None,
) -> Tuple[float, int, int]:
    """Return (avg_entropy, total_generated_tokens, total_prompts).

    Entropy is computed exclusively over generated token positions so that
    it reflects the model's output distribution peakiness during free
    generation, not its uncertainty about prompt token continuation.
    """
    LOGGER.info("Loading model from %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_vllm = vllm_config is not None and vllm_config.enabled
    if use_vllm and prompt_source.use_full_dataset:
        raise ValueError(
            "vLLM generation requires a fixed prompt list. "
            "Disable --use-full-dataset or --use-vllm."
        )

    if use_vllm:
        prompts = prompt_source.prompts or []
        if not prompts:
            raise ValueError("No prompts available for vLLM generation.")
        full_sequences, prompt_lens = generate_with_vllm(
            checkpoint,
            tokenizer,
            prompts,
            max_prompt_length,
            gen_tokens,
            vllm_config,
        )

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

        total_entropy = 0.0
        total_generated_tokens = 0
        total_prompts = 0

        total_batches = max(1, math.ceil(len(full_sequences) / batch_size))
        with torch.no_grad():
            for input_ids, attention_mask, batch_prompt_lens in tqdm(
                batch_full_sequences(full_sequences, prompt_lens, tokenizer.pad_token_id, batch_size),
                desc=f"{checkpoint.name}-vllm",
                total=total_batches,
            ):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)

                seq_len = input_ids.shape[1]
                gen_mask = torch.zeros(
                    input_ids.shape[0], seq_len, dtype=torch.bool, device=device
                )
                for i, prompt_len in enumerate(batch_prompt_lens):
                    actual_seq_len = attention_mask[i].sum().item()
                    start = max(prompt_len - 1, 0)
                    end = int(actual_seq_len) - 1
                    if end > start:
                        gen_mask[i, start:end] = True

                entropy_generated = entropy * gen_mask.float()
                total_entropy += entropy_generated.sum().item()
                total_generated_tokens += int(gen_mask.sum().item())
                total_prompts += input_ids.shape[0]

        avg_entropy = total_entropy / max(total_generated_tokens, 1)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return avg_entropy, total_generated_tokens, total_prompts

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    total_entropy = 0.0
    total_generated_tokens = 0
    total_prompts = 0

    batches, prompt_count = iter_prompt_batches(prompt_source, batch_size)

    with torch.no_grad():
        for batch in tqdm(
            list(batches) if prompt_count is not None else batches,
            desc=checkpoint.name,
        ):
            # ----------------------------------------------------------------
            # Step 1: Tokenise prompts, truncating to max_prompt_length.
            # ----------------------------------------------------------------
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].to(device)          # (B, P)
            prompt_mask = encoded["attention_mask"].to(device)    # (B, P)
            prompt_len = prompt_ids.shape[1]

            # ----------------------------------------------------------------
            # Step 2: Generate gen_tokens tokens per prompt.
            # We use greedy decoding (do_sample=False) so that generation is
            # deterministic and comparable across checkpoints. The generated
            # sequence is what we care about -- we want the entropy of the
            # distribution the model places over each generated position,
            # not just which token it picked.
            # ----------------------------------------------------------------
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            # generated shape: (B, P + gen_tokens) -- may be shorter if EOS hit

            # ----------------------------------------------------------------
            # Step 3: Forward pass on the full [prompt + generated] sequence
            # to obtain logits at every position.
            # We need logits rather than just the sampled tokens because we
            # want the full next-token distribution H = -sum_v p(v) log p(v).
            # ----------------------------------------------------------------
            full_attention_mask = (generated != tokenizer.pad_token_id).long()
            outputs = model(
                input_ids=generated,
                attention_mask=full_attention_mask,
            )
            logits = outputs.logits  # (B, P + gen_tokens, V)

            # ----------------------------------------------------------------
            # Step 4: Compute per-position entropy H over the vocabulary.
            # ----------------------------------------------------------------
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, T, V)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

            # ----------------------------------------------------------------
            # Step 5: Mask to generated positions only.
            #
            # Logit at position t predicts token t+1, so the logits that
            # correspond to the generated tokens are at positions
            # [prompt_len - 1 : prompt_len - 1 + actual_gen_len].
            # We build this mask explicitly so it handles variable-length
            # generation (e.g. if EOS was hit early for some sequences).
            # ----------------------------------------------------------------
            seq_len = generated.shape[1]
            gen_mask = torch.zeros(
                generated.shape[0], seq_len, dtype=torch.bool, device=device
            )
            for i in range(generated.shape[0]):
                # Find where generated tokens start and end (excluding padding)
                actual_seq_len = full_attention_mask[i].sum().item()
                # Logits at [prompt_len-1 .. actual_seq_len-1] predict
                # tokens at [prompt_len .. actual_seq_len], i.e. the
                # generated portion.
                start = prompt_len - 1
                end = int(actual_seq_len) - 1  # last non-pad logit position
                if end > start:
                    gen_mask[i, start:end] = True

            entropy_generated = entropy * gen_mask.float()
            total_entropy += entropy_generated.sum().item()
            total_generated_tokens += int(gen_mask.sum().item())

            # Bug fix: increment inside the loop so all batches are counted.
            total_prompts += len(batch)

    avg_entropy = total_entropy / max(total_generated_tokens, 1)

    # ------------------------------------------------------------------
    # Cleanup: free GPU memory before loading the next checkpoint.
    # Without this, loading subsequent checkpoints (especially 8B/14B)
    # will OOM because the previous model stays resident.
    # ------------------------------------------------------------------
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return avg_entropy, total_generated_tokens, total_prompts


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
    vllm_config = build_vllm_config(args)

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
            "size,lr,checkpoint,checkpoint_step,avg_entropy,num_generated_tokens,"
            "num_prompts,gen_tokens,dataset,dataset_split\n"
        )
        for ckpt in checkpoints:
            avg_entropy, num_generated_tokens, num_prompts = compute_entropy_for_checkpoint(
                ckpt.path,
                prompt_source,
                device,
                dtype,
                args.batch_size,
                args.max_prompt_length,
                args.gen_tokens,
                vllm_config,
            )
            handle.write(
                f"{ckpt.size},{ckpt.lr},{ckpt.path},"
                f"{ckpt.step if ckpt.step is not None else ''},"
                f"{avg_entropy:.6f},{num_generated_tokens},{num_prompts},"
                f"{args.gen_tokens},"
                f"{args.dataset},{args.dataset_split}\n"
            )
            handle.flush()

    LOGGER.info("Wrote results to %s", args.output_csv)


if __name__ == "__main__":
    main()