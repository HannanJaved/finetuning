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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, AutoConfig

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from .configs import SFTConfig


def _patch_deepspeed_zero3_embedding_init() -> None:
    """Work around a transformers/DeepSpeed ZeRO-3 incompatibility.

    Under `deepspeed.zero.Init()` (entered by `from_pretrained` whenever ZeRO
    stage 3 is active, regardless of accelerate's `zero3_init_flag`), a
    module's `weight.data` is a partitioned local shard, not the full tensor.
    `PreTrainedModel._init_weights`'s embedding branch does
    `module.weight.data[module.padding_idx].zero_()` without gathering the
    parameter first, so any model that sets `padding_idx` (e.g. Gemma3) hits
    `IndexError: index 0 is out of bounds for dimension 0 with size 0` on
    ranks whose local shard doesn't cover that row. Gather the parameter for
    the duration of its init, per DeepSpeed's documented pattern for custom
    weight init under ZeRO-3.
    """
    from transformers import modeling_utils
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if getattr(modeling_utils.PreTrainedModel._init_weights, "_zero3_embedding_patched", False):
        return

    original_init_weights = modeling_utils.PreTrainedModel._init_weights

    def patched_init_weights(self, module):
        if (
            isinstance(module, torch.nn.Embedding)
            and module.padding_idx is not None
            and is_deepspeed_zero3_enabled()
        ):
            import deepspeed

            with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    original_init_weights(self, module)
        else:
            original_init_weights(self, module)

    patched_init_weights._zero3_embedding_patched = True
    modeling_utils.PreTrainedModel._init_weights = patched_init_weights


_patch_deepspeed_zero3_embedding_init()


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    # Load the model config so we can set attributes (like use_cache) that
    # some custom model classes expect to find on the config object instead
    # of being passed as a kwarg to __init__.
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Gemma-3-4B-pt (and similar) ship as multimodal Gemma3Config
    # (Gemma3ForConditionalGeneration). AutoModelForCausalLM then loads the
    # VLM wrapper, and TRL DPO treats model_type "gemma3" as vision-only
    # (process_row + processing_class.tokenizer), which breaks text-only DPO
    # when we pass a plain tokenizer. Load the text backbone as
    # Gemma3ForCausalLM instead so TRL uses tokenize_row.
    if getattr(config, "model_type", None) == "gemma3" and hasattr(config, "text_config"):
        config = config.text_config

    # Some model implementations (e.g. custom Gemma3 in this environment)
    # don't accept `use_cache` as an __init__ kwarg. Set it on the config
    # instead to control caching behaviour during generation/training.
    config.use_cache = False if training_args.gradient_checkpointing else True

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        config=config,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    return model
