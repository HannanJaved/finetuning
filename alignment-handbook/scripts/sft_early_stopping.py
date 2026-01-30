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
Supervised fine-tuning script with early stopping support.
Automatically creates a validation split if not present in the dataset.

Usage:
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft_early_stopping.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 true \
    --eval_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --early_stopping_patience 3 \
    --output_dir data/Qwen2.5-1.5B-SFT
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

from src.alignment import ScriptArguments, SFTConfig, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format


## Fix timeout issue for long preproc times
import torch
orig_init = torch.distributed.init_process_group

def patched_init(*args, **kwargs):
    from datetime import timedelta
    kwargs['timeout'] = timedelta(hours=1)
    return orig_init(*args, **kwargs)

torch.distributed.init_process_group = patched_init
## end fix

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

    ################
    # Load datasets
    ################
    dataset = get_dataset(script_args)
    
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
    
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    
    ############
    # Load model
    ############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

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

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[train_split],
        eval_dataset=(dataset[test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=callbacks if callbacks else None,
    )

    ###############
    # Training loop
    ###############
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

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
