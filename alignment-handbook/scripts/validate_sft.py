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
SFT Validation script - computes validation loss on a trained SFT checkpoint.

Usage:
accelerate launch scripts/validate_sft.py --config validation_sft_config.yaml
"""

import json
import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import set_seed

from src.alignment import ScriptArguments, SFTConfig, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format


## Fix timeout issue for long preproc times
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

    ################
    # Load datasets
    ################
    dataset = get_dataset(script_args)
    
    # Create validation split if no test split exists
    if script_args.dataset_test_split not in dataset:
        logger.info(f"No test split '{script_args.dataset_test_split}' found. Creating validation split from training data.")
        train_data = dataset[script_args.dataset_train_split]
        split_dataset = train_data.train_test_split(test_size=0.05, seed=42)
        dataset = datasets.DatasetDict({
            script_args.dataset_train_split: split_dataset["train"],
            script_args.dataset_test_split: split_dataset["test"]
        })
        logger.info(f"Created validation split with {len(dataset[script_args.dataset_test_split])} samples")
    
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
    # Initialize the SFT Trainer
    ############################
    eval_dataset = dataset[script_args.dataset_test_split]
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=eval_dataset,  # Required for initialization, but won't be used
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ##########
    # Evaluate
    ##########
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # Also save to a dedicated results file
    results = {
        "mode": "sft",
        "model_path": model_args.model_name_or_path,
        "dataset_name": script_args.dataset_name,
        "metrics": metrics
    }
    
    results_file = os.path.join(training_args.output_dir, "validation_results.json")
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"*** Validation Results ***")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
