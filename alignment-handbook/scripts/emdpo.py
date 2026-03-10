#!/usr/bin/env python
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.

"""EM-style latent reliability DPO on the original fixed preference dataset."""

import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import DPOConfig, EMDPOTrainer, ScriptArguments, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info("Model parameters %s", model_args)
    logger.info("Script parameters %s", script_args)
    logger.info("Training parameters %s", training_args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info("Checkpoint detected, resuming training at %s.", last_checkpoint)

    model = get_model(model_args, training_args)
    ref_model = get_model(model_args, training_args)
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    dataset = get_dataset(script_args)
    for split in dataset:
        cols_to_remove = []
        if "messages" in dataset[split].column_names:
            cols_to_remove.append("messages")
        if "prompt" in dataset[split].column_names:
            cols_to_remove.append("prompt")
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

        dataset[split] = dataset[split].map(
            _strip_conversation,
            desc=f"Stripping metadata from {split} conversations",
        )

    trainer = EMDPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    logger.info("*** Train EM-DPO ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if not training_args.emdpo_enabled:
        logger.warning("EM-DPO script invoked with emdpo_enabled=False; behavior may not match intended setup.")
    main(script_args, training_args, model_args)