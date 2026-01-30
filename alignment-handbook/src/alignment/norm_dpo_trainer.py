# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from trl import DPOTrainer


class NormDPOTrainer(DPOTrainer):
    """Direct Preference Optimization trainer with optional length-normalized log-probabilities.

    This class mirrors :class:`trl.DPOTrainer` but, when enabled via config, divides policy and
    reference log-prob sums by their respective completion lengths before computing the DPO loss.
    """

    _name = "NormDPO"

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        args: Any | None = None,
        **kwargs: Any,
    ) -> None:
        # fall back to base class default config if none supplied
        super().__init__(model=model, ref_model=ref_model, args=args, **kwargs)

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        loss_type: str = "sigmoid",
        model_output: dict[str, torch.FloatTensor] | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if getattr(self.args, "length_normalize_logps", False):
            (
                chosen_logps,
                rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            ) = self._apply_length_normalization(
                chosen_logps,
                rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                model_output,
            )

        return super().dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            loss_type,
            model_output,
        )

    def _apply_length_normalization(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        model_output: dict[str, torch.FloatTensor] | None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if model_output is None:
            raise ValueError("Length normalization requires the policy forward output (model_output).")
        if "chosen_lengths" not in model_output or "rejected_lengths" not in model_output:
            raise ValueError(
                "Model output must include `chosen_lengths` and `rejected_lengths` for length normalization;"
                " upgrade TRL or ensure the trainer exposes these fields."
            )

        denom_chosen, denom_rejected = self._get_length_denominators(
            model_output, chosen_logps.device, chosen_logps.dtype
        )

        if not model_output.get("_length_normalized", False):
            chosen_logps = chosen_logps / denom_chosen
            rejected_logps = rejected_logps / denom_rejected
            model_output["chosen_logps"] = chosen_logps
            model_output["rejected_logps"] = rejected_logps
            model_output["_length_normalized"] = True
        else:
            chosen_logps = model_output["chosen_logps"]
            rejected_logps = model_output["rejected_logps"]

        if not self.reference_free:
            ref_chosen_logps = ref_chosen_logps.to(denom_chosen.device, dtype=chosen_logps.dtype) / denom_chosen
            ref_rejected_logps = ref_rejected_logps.to(denom_rejected.device, dtype=rejected_logps.dtype) / denom_rejected

        return chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps

    def _get_length_denominators(
        self,
        model_output: dict[str, torch.FloatTensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if "_length_norm_denominator_chosen" in model_output and "_length_norm_denominator_rejected" in model_output:
            return (
                model_output["_length_norm_denominator_chosen"].to(device=device, dtype=dtype),
                model_output["_length_norm_denominator_rejected"].to(device=device, dtype=dtype),
            )

        chosen_lengths = model_output["chosen_lengths"].to(device=device, dtype=dtype)
        rejected_lengths = model_output["rejected_lengths"].to(device=device, dtype=dtype)

        denom_chosen = self._build_length_denominator(chosen_lengths)
        denom_rejected = self._build_length_denominator(rejected_lengths)

        model_output["_length_norm_denominator_chosen"] = denom_chosen
        model_output["_length_norm_denominator_rejected"] = denom_rejected

        return denom_chosen, denom_rejected

    def _build_length_denominator(self, lengths: torch.FloatTensor) -> torch.FloatTensor:
        denom = torch.clamp(lengths, min=float(getattr(self.args, "length_norm_min_tokens", 1.0)))
        exponent = float(getattr(self.args, "length_norm_exponent", 1.0))
        if exponent != 1.0:
            denom = denom.pow(exponent)
        return denom

    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        batch: dict[str, list | torch.LongTensor],
        train_eval: str = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Override to attach completion lengths so length normalization works with stock TRL outputs."""

        if getattr(self.args, "use_liger_kernel", False):
            # Defer to base for the Liger path; length-normalization is not wired for fused kernel.
            return super().get_batch_loss_metrics(model, batch, train_eval)

        metrics = {}
        model_output = self.concatenated_forward(model, batch)

        # Compute completion lengths from attention masks; this stays in sync with truncation/padding done upstream.
        completion_attention_mask = batch.get("chosen_attention_mask")
        rejected_attention_mask = batch.get("rejected_attention_mask")
        if completion_attention_mask is not None and rejected_attention_mask is not None:
            model_output["chosen_lengths"] = completion_attention_mask.sum(dim=1)
            model_output["rejected_lengths"] = rejected_attention_mask.sum(dim=1)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses = 0
        chosen_rewards = 0
        rejected_rewards = 0

        for idx, loss_type in enumerate(self.loss_type):
            _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                model_output["chosen_logps"],
                model_output["rejected_logps"],
                ref_chosen_logps,
                ref_rejected_logps,
                loss_type,
                model_output,
            )

            weight = self.loss_weights[idx] if self.loss_weights else 1.0
            losses = losses + _losses * weight
            chosen_rewards = chosen_rewards + _chosen_rewards * weight
            rejected_rewards = rejected_rewards + _rejected_rewards * weight

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics
