# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DPO training script with early stopping support.
Automatically creates a validation split if not present in the dataset.

Usage:
python scripts/dpo_early_stopping.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --early_stopping_patience 3 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
"""

import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import set_seed, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

from alignment import DPOConfig, ScriptArguments, get_dataset, get_model, get_tokenizer
from trl import DPOTrainer, ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
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

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###################
    # Model & Tokenizer
    ###################
    model = get_model(model_args, training_args)
    ref_model = get_model(model_args, training_args)
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    #########
    # Dataset
    #########
    dataset = get_dataset(script_args)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    ####################################
    # Create validation split if needed
    ####################################
    train_split = script_args.dataset_train_split
    test_split = script_args.dataset_test_split
    
    # Check if we need evaluation and if test split exists
    if training_args.eval_strategy != "no":
        if test_split not in dataset:
            logger.info(f"No '{test_split}' split found. Creating validation split from training data.")
            # Get validation ratio from environment or use default 5%
            val_ratio = float(os.environ.get("VAL_SPLIT_RATIO", "0.05"))
            val_seed = int(os.environ.get("VAL_SPLIT_SEED", "42"))
            
            # Split the training data
            split_dataset = dataset[train_split].train_test_split(
                test_size=val_ratio,
                seed=val_seed,
                shuffle=True
            )
            dataset[train_split] = split_dataset["train"]
            dataset[test_split] = split_dataset["test"]
            
            logger.info(f"Created validation split: {len(dataset[test_split])} samples ({val_ratio*100:.1f}%)")
            logger.info(f"Remaining training samples: {len(dataset[train_split])}")

    ############################
    # Setup early stopping
    ############################
    callbacks = []
    early_stopping_patience = int(training_args.early_stopping_patience if training_args.early_stopping_patience is not None else 0)
    early_stopping_threshold = float(training_args.early_stopping_threshold if training_args.early_stopping_threshold is not None else 0.0)
    
    if early_stopping_patience > 0 and training_args.eval_strategy != "no":
        logger.info(f"Enabling early stopping with patience={early_stopping_patience}, threshold={early_stopping_threshold}")
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
        )

    ##########
    # Training
    ##########
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[train_split],
        eval_dataset=dataset[test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=callbacks if callbacks else None,
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
