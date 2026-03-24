"""Reference-Anchored SFT training script."""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.alignment import RASFTTrainer, ScriptArguments, SFTConfig, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, TrlParser, get_peft_config, setup_chat_format

import torch
orig_init = torch.distributed.init_process_group


def patched_init(*args, **kwargs):
    from datetime import timedelta
    kwargs["timeout"] = timedelta(hours=1)
    return orig_init(*args, **kwargs)


torch.distributed.init_process_group = patched_init

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

    dataset = get_dataset(script_args)

    def has_valid_messages(example):
        messages = example.get("messages")
        if messages is None or not isinstance(messages, list):
            return False
        for message in messages:
            if message is None or not isinstance(message, dict):
                return False
            if message.get("content") is None:
                return False
        return True

    if isinstance(dataset, datasets.DatasetDict):
        for split_name in dataset.keys():
            dataset[split_name] = dataset[split_name].filter(has_valid_messages)
    else:
        dataset = dataset.filter(has_valid_messages)

    max_chars_per_sample = int(getattr(training_args, "rasft_max_chars_per_sample", 200000))

    def within_length_budget(example):
        messages = example.get("messages") or []
        total_chars = 0
        for message in messages:
            content = message.get("content", "") if isinstance(message, dict) else ""
            if content is not None:
                total_chars += len(content)
            if total_chars > max_chars_per_sample:
                return False
        return True

    if isinstance(dataset, datasets.DatasetDict):
        for split_name in dataset.keys():
            before_count = len(dataset[split_name])
            dataset[split_name] = dataset[split_name].filter(
                within_length_budget,
                desc=f"Filtering long samples from {split_name}",
            )
            after_count = len(dataset[split_name])
            logger.info(
                "Filtered %s long samples from %s using rasft_max_chars_per_sample=%s",
                before_count - after_count,
                split_name,
                max_chars_per_sample,
            )
    else:
        before_count = len(dataset)
        dataset = dataset.filter(within_length_budget, desc="Filtering long samples")
        after_count = len(dataset)
        logger.info(
            "Filtered %s long samples using rasft_max_chars_per_sample=%s",
            before_count - after_count,
            max_chars_per_sample,
        )

    tokenizer = get_tokenizer(model_args, training_args)
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    eval_dataset = None
    if training_args.eval_strategy != "no":
        if script_args.dataset_test_split in dataset:
            eval_dataset = dataset[script_args.dataset_test_split]
        else:
            logger.warning(
                "Eval requested but dataset split '%s' is missing. Set dataset_test_split_size to create one or disable eval.",
                script_args.dataset_test_split,
            )

    trainer = RASFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        model_args=model_args,
    )

    logger.info("*** Train RA-SFT ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    if script_args.dataset_train_split in dataset:
        metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)

    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
        "tags": ["alignment-handbook", "ra-sft"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        if script_args.dataset_test_split in dataset:
            metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if not training_args.rasft_enabled:
        logger.warning("RA-SFT script invoked with rasft_enabled=False; behavior may reduce to standard SFT.")
    main(script_args, training_args, model_args)
