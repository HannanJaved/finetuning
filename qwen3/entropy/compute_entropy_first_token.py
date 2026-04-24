#!/usr/bin/env python3
"""Compute entropy only at the first generated token for SFT checkpoints.

This script mirrors compute_entropy.py but averages entropy across the *first* generated
position per prompt. It writes a CSV with the same schema as compute_entropy.py so
existing plotting utilities can be reused.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import compute_entropy

LOGGER = logging.getLogger(__name__)


def compute_first_token_entropy_for_checkpoint(
    checkpoint: Path,
    prompt_source: compute_entropy.PromptSource,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_prompt_length: int,
    gen_tokens: int,
    vllm_config: compute_entropy.VllmConfig | None = None,
) -> tuple[float, int, int]:
    """Return (avg_entropy, total_first_tokens, total_prompts)."""
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
        full_sequences, prompt_lens = compute_entropy.generate_with_vllm(
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
        total_first_tokens = 0
        total_prompts = 0

        total_batches = max(1, math.ceil(len(full_sequences) / batch_size))
        with torch.no_grad():
            for input_ids, attention_mask, batch_prompt_lens in tqdm(
                compute_entropy.batch_full_sequences(
                    full_sequences, prompt_lens, tokenizer.pad_token_id, batch_size
                ),
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

                for i, prompt_len in enumerate(batch_prompt_lens):
                    actual_seq_len = int(attention_mask[i].sum().item())
                    if actual_seq_len <= prompt_len:
                        continue
                    index = max(prompt_len - 1, 0)
                    total_entropy += entropy[i, index].item()
                    total_first_tokens += 1

                total_prompts += input_ids.shape[0]

        avg_entropy = total_entropy / max(total_first_tokens, 1)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return avg_entropy, total_first_tokens, total_prompts

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    total_entropy = 0.0
    total_first_tokens = 0
    total_prompts = 0

    batches, prompt_count = compute_entropy.iter_prompt_batches(prompt_source, batch_size)

    with torch.no_grad():
        for batch in tqdm(
            list(batches) if prompt_count is not None else batches,
            desc=checkpoint.name,
        ):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].to(device)
            prompt_mask = encoded["attention_mask"].to(device)

            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            full_attention_mask = (generated != tokenizer.pad_token_id).long()
            outputs = model(
                input_ids=generated,
                attention_mask=full_attention_mask,
            )
            logits = outputs.logits

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)

            for i in range(generated.shape[0]):
                prompt_len = int(prompt_mask[i].sum().item())
                actual_seq_len = int(full_attention_mask[i].sum().item())
                if actual_seq_len <= prompt_len:
                    continue
                index = max(prompt_len - 1, 0)
                total_entropy += entropy[i, index].item()
                total_first_tokens += 1

            total_prompts += len(batch)

    avg_entropy = total_entropy / max(total_first_tokens, 1)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return avg_entropy, total_first_tokens, total_prompts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = compute_entropy.parse_args()

    if args.output_csv == Path("entropy_results.csv"):
        args.output_csv = Path("entropy_first_token_results.csv")

    device = compute_entropy.resolve_device(args.device)
    dtype = compute_entropy.resolve_dtype(args.dtype, device)
    vllm_config = compute_entropy.build_vllm_config(args)

    prompt_source = compute_entropy.load_prompt_source(
        args.dataset,
        args.dataset_config,
        args.dataset_split,
        args.num_prompts,
        args.prompt_fraction,
        args.seed,
        args.probe_cache,
        args.use_full_dataset,
    )

    checkpoints = compute_entropy.build_checkpoint_index(
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
            avg_entropy, num_first_tokens, num_prompts = compute_first_token_entropy_for_checkpoint(
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
                f"{avg_entropy:.6f},{num_first_tokens},{num_prompts},"
                f"{args.gen_tokens},"
                f"{args.dataset},{args.dataset_split}\n"
            )
            handle.flush()

    LOGGER.info("Wrote results to %s", args.output_csv)


if __name__ == "__main__":
    main()
