# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from trl import SFTTrainer


class RASFTTrainer(SFTTrainer):
    """Reference-Anchored SFT with token weights from current-policy entropy and base-model novelty."""

    def __init__(self, *args, model_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rasft_base_model = None
        self._rasft_model_args = model_args
        self._maybe_init_base_model()

    def _maybe_init_base_model(self) -> None:
        if not bool(getattr(self.args, "rasft_enabled", False)):
            return

        base_model_name = getattr(self.args, "rasft_base_model_name_or_path", None)
        if base_model_name is None and self._rasft_model_args is not None:
            base_model_name = getattr(self._rasft_model_args, "model_name_or_path", None)
        if base_model_name is None:
            raise ValueError("RA-SFT requires `rasft_base_model_name_or_path` or an available model_name_or_path.")

        torch_dtype = getattr(self._rasft_model_args, "torch_dtype", None) if self._rasft_model_args is not None else None
        attn_impl = getattr(self._rasft_model_args, "attn_implementation", None) if self._rasft_model_args is not None else None
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype not in (None, "auto"):
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        self._rasft_base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        self._rasft_base_model.eval()
        self._rasft_base_model.requires_grad_(False)
        self._move_model_to_device(self._rasft_base_model, self.accelerator.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not bool(getattr(self.args, "rasft_enabled", False)):
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        if self._rasft_base_model is None:
            self._maybe_init_base_model()

        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if labels is None or input_ids is None:
            raise ValueError("RA-SFT requires tokenized `input_ids` and `labels`.")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)
        target_ids = shift_labels.masked_fill(~valid_mask, 0)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)

        token_nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none",
        ).view_as(target_ids)

        if bool(getattr(self.args, "rasft_entropy_normalize", True)):
            vocab_size = logits.size(-1)
            entropy = entropy / max(math.log(vocab_size), 1.0)

        with torch.no_grad():
            base_outputs = self._rasft_base_model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_outputs.logits[:, :-1, :]
            base_log_probs = F.log_softmax(base_logits, dim=-1)
            base_token_probs = base_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1).exp()

        weights = entropy * (1.0 - base_token_probs)
        weight_min = float(getattr(self.args, "rasft_weight_min", 0.0))
        weight_max = float(getattr(self.args, "rasft_weight_max", 1.0))
        weights = weights.clamp(min=weight_min, max=weight_max)
        weights = weights * valid_mask.to(weights.dtype)

        denom = weights.sum().clamp_min(1e-8)
        loss = (weights * token_nll).sum() / denom

        with torch.no_grad():
            valid_weights = weights[valid_mask]
            valid_entropy = entropy[valid_mask]
            valid_base_probs = base_token_probs[valid_mask]
            metrics = {
                "rasft/weight_mean": valid_weights.mean().item() if valid_mask.any() else 0.0,
                "rasft/weight_max": valid_weights.max().item() if valid_mask.any() else 0.0,
                "rasft/entropy_mean": valid_entropy.mean().item() if valid_mask.any() else 0.0,
                "rasft/base_prob_mean": valid_base_probs.mean().item() if valid_mask.any() else 0.0,
                "rasft/token_nll_mean": token_nll[valid_mask].mean().item() if valid_mask.any() else 0.0,
            }
            self.log(metrics)

        return (loss, outputs) if return_outputs else loss
