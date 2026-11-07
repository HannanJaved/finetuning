"""
Batch DPO validation: same computation as scripts/validate_dpo.py (same dataset preprocessing,
same 0.0005 test split with seed dataset_test_split_seed, DPOTrainer.evaluate()), but the
reference model is loaded once and the script loops over many policy checkpoints.

Manifest: TSV with columns  model_path <TAB> beta <TAB> output_dir  (one checkpoint per line).
Each checkpoint writes <output_dir>/validation_results.json in the validate_dpo.py format.
Checkpoints whose results file already exists are skipped, so jobs can be resubmitted.

Usage (single GPU):
python scripts/validate_dpo_batch.py --config <dpo yaml for this model size> \
    --ref_model_name_or_path <SFT checkpoint> --manifest <tsv> --shard 0 --num_shards 4 --report_to none
"""

import gc
import json
import logging
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from transformers import set_seed

from alignment import DPOConfig, ScriptArguments, get_dataset, get_model, get_tokenizer
from trl import DPOTrainer, ModelConfig, TrlParser

logger = logging.getLogger(__name__)


@dataclass
class BatchArgs:
    manifest: str = field(default=None, metadata={"help": "TSV: model_path, beta, output_dir"})
    shard: int = field(default=0)
    num_shards: int = field(default=1)


def load_eval_dataset(script_args):
    """Identical to the dataset block of scripts/validate_dpo.py."""
    dataset = get_dataset(script_args)
    for split in dataset:
        cols_to_remove = [c for c in ("messages", "prompt") if c in dataset[split].column_names]
        if cols_to_remove:
            dataset[split] = dataset[split].remove_columns(cols_to_remove)

        def _strip_conversation(example):
            for key in ["chosen", "rejected"]:
                if key in example and isinstance(example[key], list):
                    example[key] = [
                        {"role": turn["role"], "content": turn["content"]}
                        for turn in example[key]
                        if isinstance(turn, dict) and "role" in turn and "content" in turn
                    ]
            return example

        dataset[split] = dataset[split].map(_strip_conversation, desc=f"Stripping metadata from {split}")

    if script_args.dataset_test_split not in dataset:
        train_data = dataset[script_args.dataset_train_split]
        test_split_size = script_args.dataset_test_split_size or 0.0005
        split_dataset = train_data.train_test_split(test_size=test_split_size, seed=script_args.dataset_test_split_seed)
        dataset = datasets.DatasetDict({
            script_args.dataset_train_split: split_dataset["train"],
            script_args.dataset_test_split: split_dataset["test"],
        })
    return dataset[script_args.dataset_test_split]


def main(script_args, training_args, model_args, batch_args):
    set_seed(training_args.seed)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO)
    transformers.utils.logging.set_verbosity_warning()
    if training_args.length_normalize_logps:
        raise ValueError("Batch validation only supports the standard sigmoid DPO loss.")
    if not script_args.ref_model_name_or_path:
        raise ValueError("--ref_model_name_or_path is required (the SFT checkpoint DPO started from).")

    with open(batch_args.manifest) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    rows = rows[batch_args.shard::batch_args.num_shards]
    todo = [r for r in rows if not os.path.exists(os.path.join(r[2], "validation_results.json"))]
    logger.info(f"Shard {batch_args.shard}/{batch_args.num_shards}: {len(rows)} checkpoints, {len(todo)} to do")
    if not todo:
        return

    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_dataset = load_eval_dataset(script_args)

    ref_model_args = deepcopy(model_args)
    ref_model_args.model_name_or_path = script_args.ref_model_name_or_path
    ref_model = get_model(ref_model_args, training_args)

    for model_path, beta, output_dir in todo:
        logger.info(f"Evaluating {model_path} (beta={beta})")
        policy_args = deepcopy(model_args)
        policy_args.model_name_or_path = model_path
        args = deepcopy(training_args)
        args.beta = float(beta)
        args.output_dir = output_dir
        model = get_model(policy_args, args)
        trainer = DPOTrainer(model, ref_model, args=args, train_dataset=eval_dataset,
                             eval_dataset=eval_dataset, processing_class=tokenizer)
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        results = {
            "mode": "dpo",
            "model_path": model_path,
            "ref_model_path": script_args.ref_model_name_or_path,
            "beta": float(beta),
            "dataset_name": script_args.dataset_name,
            "metrics": metrics,
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"eval_loss={metrics['eval_loss']:.4f} -> {output_dir}")
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig, BatchArgs))
    main(*parser.parse_args_and_config())
