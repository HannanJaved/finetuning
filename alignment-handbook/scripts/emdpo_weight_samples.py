"""Score each example in the training set with EMDPO weights and save high/low samples.

Usage (mirrors the shell script):
    python scripts/emdpo_weight_samples.py \
        --config path/to/config.yaml \
        --sample_output_dir /path/to/out \
        --sample_size 10 \
        --dataset_fraction 0.1
"""

import argparse
import json
import logging
import os
import sys

import datasets
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import DPOConfig, EMDPOTrainer, ScriptArguments, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, TrlParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", required=True, help="Path to the YAML training config.")
    base.add_argument("--sample_output_dir", required=True, help="Directory to write sample JSON files.")
    base.add_argument("--sample_size", type=int, default=10, help="Number of high/low weight examples to save.")
    base.add_argument("--dataset_fraction", type=float, default=1.0,
                      help="Randomly subsample this fraction of the dataset before scoring (e.g. 0.1 for 10%%).")
    known, remaining = base.parse_known_args()
    return known, remaining


# ---------------------------------------------------------------------------
# Weight scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_dataset(trainer: EMDPOTrainer, dataloader: DataLoader, device: torch.device):
    """Iterate over all batches and collect per-example EMDPO weights + diagnostics."""
    all_weights = []
    all_diags = []
    all_indices = []

    total = len(dataloader)
    global_idx = 0
    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass through policy
        model_output = trainer.concatenated_forward(trainer.model, batch)

        # Attach lengths (same logic as in get_batch_loss_metrics)
        chosen_mask = batch.get("chosen_attention_mask")
        rejected_mask = batch.get("rejected_attention_mask")
        if chosen_mask is not None and rejected_mask is not None:
            model_output["chosen_lengths"] = chosen_mask.sum(dim=1)
            model_output["rejected_lengths"] = rejected_mask.sum(dim=1)

        # Reference log-probs
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"].to(device)
            ref_rejected_logps = batch["ref_rejected_logps"].to(device)
        else:
            ref_chosen_logps, ref_rejected_logps = trainer.compute_ref_log_probs(batch)

        trainer._current_model_output = model_output
        try:
            weights, diags = trainer._compute_emdpo_weights(
                model_output["chosen_logps"],
                model_output["rejected_logps"],
                ref_chosen_logps,
                ref_rejected_logps,
            )
        finally:
            trainer._current_model_output = None

        bsz = weights.shape[0]
        all_weights.append(weights.cpu())
        all_diags.append({k: v.cpu() for k, v in diags.items()})
        all_indices.extend(range(global_idx, global_idx + bsz))
        global_idx += bsz
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total:
            logger.info("  scored %d / %d batches (%d examples)", batch_idx + 1, total, global_idx)

    weights_tensor = torch.cat(all_weights, dim=0)          # (N,)
    merged_diags = {
        k: torch.cat([d[k] for d in all_diags], dim=0)
        for k in all_diags[0]
    }
    return weights_tensor, merged_diags, all_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_conversation(token_ids, tokenizer):
    """Best-effort decode; skip special tokens for readability."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    # Strip padding (token_id == tokenizer.pad_token_id)
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        token_ids = [t for t in token_ids if t != pad_id]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _build_record(dataset_example, weight: float, diags: dict, tokenizer):
    """Build a JSON-serialisable dict for one example."""
    record = {"weight": round(float(weight), 6)}

    # Include raw text fields if available (dataset may have them)
    for key in ("chosen", "rejected", "prompt"):
        if key in dataset_example:
            val = dataset_example[key]
            if isinstance(val, list):
                # chat format: list of dicts with role/content
                val = [
                    {"role": t.get("role", ""), "content": t.get("content", "")}
                    for t in val
                    if isinstance(t, dict)
                ]
            record[key] = val

    record["diagnostics"] = {k: round(float(v), 6) for k, v in diags.items()}
    return record


def save_samples(samples: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d samples to %s", len(samples), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cli_args, remaining = parse_args()

    # Inject --config into sys.argv so TrlParser can pick it up
    sys.argv = [sys.argv[0]] + remaining + ["--config", cli_args.config]

    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())

    os.makedirs(cli_args.sample_output_dir, exist_ok=True)

    # ref_model  = model_name_or_path  (post-SFT, the DPO starting point)
    # policy     = last checkpoint in output_dir  (post-DPO)
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    ref_model = get_model(model_args, training_args)
    logger.info("Loaded reference model from: %s", model_args.model_name_or_path)

    if last_checkpoint is not None:
        logger.info("Loading trained policy from checkpoint: %s", last_checkpoint)
        import copy
        model_args_for_policy = copy.copy(model_args)
        object.__setattr__(model_args_for_policy, "model_name_or_path", last_checkpoint)
        model = get_model(model_args_for_policy, training_args)
    else:
        logger.info(
            "No checkpoint found in output_dir=%s; using model_name_or_path as the policy.",
            training_args.output_dir,
        )
        model = get_model(model_args, training_args)
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(script_args)
    train_dataset = dataset[script_args.dataset_train_split]

    if cli_args.dataset_fraction < 1.0:
        n_keep = max(1, int(len(train_dataset) * cli_args.dataset_fraction))
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(n_keep))
        logger.info("Subsampled dataset to %d examples (%.1f%%)", n_keep, cli_args.dataset_fraction * 100)

    # Strip metadata columns (same as emdpo.py)
    cols_to_remove = [c for c in ("messages", "prompt") if c in train_dataset.column_names]
    if cols_to_remove:
        train_dataset = train_dataset.remove_columns(cols_to_remove)

    def _strip_conversation(example):
        for key in ["chosen", "rejected"]:
            if key in example and isinstance(example[key], list):
                example[key] = [
                    {"role": t["role"], "content": t["content"]}
                    for t in example[key]
                    if isinstance(t, dict) and "role" in t and "content" in t
                ]
        return example

    train_dataset = train_dataset.map(_strip_conversation, desc="Stripping metadata")

    # Build trainer (handles tokenisation + data collation internally)
    trainer = EMDPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    device = next(model.parameters()).device
    model.eval()
    if ref_model is not None:
        ref_model.eval()

    logger.info("Scoring %d examples…", len(train_dataset))
    weights, diags, indices = score_dataset(trainer, trainer.get_train_dataloader(), device)

    n = cli_args.sample_size
    sorted_order = torch.argsort(weights)          # ascending → low first
    low_indices = sorted_order[:n].tolist()
    high_indices = sorted_order[-n:].flip(0).tolist()

    def build_records(idx_list):
        records = []
        for rank, idx in enumerate(idx_list):
            dataset_idx = indices[idx]
            example = train_dataset[dataset_idx]
            per_diag = {k: v[idx].item() for k, v in diags.items()}
            record = _build_record(example, weights[idx].item(), per_diag, tokenizer)
            record["dataset_index"] = dataset_idx
            record["rank"] = rank + 1
            records.append(record)
        return records

    low_records = build_records(low_indices)
    high_records = build_records(high_indices)

    save_samples(low_records, os.path.join(cli_args.sample_output_dir, "low_weight_samples.json"))
    save_samples(high_records, os.path.join(cli_args.sample_output_dir, "high_weight_samples.json"))

    # Also save a brief weight distribution summary
    summary = {
        "total_examples": len(weights),
        "weight_mean": round(weights.mean().item(), 6),
        "weight_std": round(weights.std().item(), 6),
        "weight_min": round(weights.min().item(), 6),
        "weight_max": round(weights.max().item(), 6),
        "weight_p10": round(weights.quantile(0.10).item(), 6),
        "weight_p25": round(weights.quantile(0.25).item(), 6),
        "weight_p50": round(weights.quantile(0.50).item(), 6),
        "weight_p75": round(weights.quantile(0.75).item(), 6),
        "weight_p90": round(weights.quantile(0.90).item(), 6),
    }
    save_samples([summary], os.path.join(cli_args.sample_output_dir, "weight_distribution.json"))

    logger.info("Done. Outputs written to %s", cli_args.sample_output_dir)


if __name__ == "__main__":
    main()
