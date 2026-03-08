# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from .norm_dpo_trainer import NormDPOTrainer


class LNDPOV2Trainer(NormDPOTrainer):
    """LN-DPO v2 with a bounded, length-normalized reliability signal.

    Compared with the original LN-DPO heuristic, this version avoids using raw
    reference log-prob sums directly as the confidence signal. Instead it builds
    a bounded margin from length-normalized reference log-probs, which reduces
    response-length bias and prevents very large-magnitude margins from collapsing
    the reliability estimator into a near-global damping term.
    """

    _name = "LNDPOV2"

    def _compute_lndpo_weights(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        model_output = getattr(self, "_current_model_output", None)
        if model_output is None:
            raise ValueError("LN-DPO v2 requires cached model_output to compute normalized reliability margins.")

        if not self.reference_free:
            ref_chosen_logps, ref_rejected_logps = self._apply_length_normalization(
                chosen_logps=chosen_logps,
                rejected_logps=rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                model_output=model_output,
            )[2:]

        ref_margin = (ref_chosen_logps - ref_rejected_logps).to(chosen_logps.device, dtype=chosen_logps.dtype)

        margin_clip = float(getattr(self.args, "lndpo_v2_margin_clip", 5.0))
        bounded_margin = torch.clamp(ref_margin, min=-margin_clip, max=margin_clip)

        scale = float(getattr(self.args, "lndpo_v2_margin_scale", 2.0))
        scaled_ref_margin = bounded_margin * scale

        noise_eps = float(getattr(self.args, "lndpo_noise_eps", 0.1))
        noise_eps = min(max(noise_eps, 0.0), 0.499)

        posterior_clean = (1.0 - noise_eps) * torch.sigmoid(scaled_ref_margin) + noise_eps * torch.sigmoid(
            -scaled_ref_margin
        )
        posterior_clean = torch.clamp(posterior_clean, min=1e-6, max=1.0)

        min_weight = float(getattr(self.args, "lndpo_min_weight", 0.2))
        max_weight = float(getattr(self.args, "lndpo_max_weight", 1.0))
        reliability_weight = torch.clamp(posterior_clean, min=min_weight, max=max_weight)

        if getattr(self.args, "lndpo_detach_weights", True):
            reliability_weight = reliability_weight.detach()

        return reliability_weight, scaled_ref_margin.detach()

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        if getattr(self.args, "use_liger_kernel", False):
            return super().get_batch_loss_metrics(model, batch, train_eval)

        self._current_model_output = None
        try:
            metrics = {}
            model_output = self.concatenated_forward(model, batch)
            self._current_model_output = model_output

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
            lndpo_weights = None
            lndpo_ref_margin = None

            lndpo_enabled = bool(getattr(self.args, "lndpo_v2_enabled", False))
            lndpo_warmup_steps = int(getattr(self.args, "lndpo_warmup_steps", 0))
            if lndpo_enabled and self.state.global_step >= lndpo_warmup_steps:
                lndpo_weights, lndpo_ref_margin = self._compute_lndpo_weights(
                    model_output["chosen_logps"],
                    model_output["rejected_logps"],
                    ref_chosen_logps,
                    ref_rejected_logps,
                )

            for idx, loss_type in enumerate(self.loss_type):
                _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                    model_output["chosen_logps"],
                    model_output["rejected_logps"],
                    ref_chosen_logps,
                    ref_rejected_logps,
                    loss_type,
                    model_output,
                )

                if lndpo_weights is not None:
                    _losses = _losses * lndpo_weights

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
            if lndpo_weights is not None:
                metrics[f"{prefix}lndpo_v2/weight_mean"] = (
                    self.accelerator.gather_for_metrics(lndpo_weights).detach().mean().item()
                )
                metrics[f"{prefix}lndpo_v2/weight_min"] = (
                    self.accelerator.gather_for_metrics(lndpo_weights).detach().min().item()
                )
                metrics[f"{prefix}lndpo_v2/weight_max"] = (
                    self.accelerator.gather_for_metrics(lndpo_weights).detach().max().item()
                )
                metrics[f"{prefix}lndpo_v2/ref_margin_mean"] = (
                    self.accelerator.gather_for_metrics(lndpo_ref_margin).detach().mean().item()
                )

            return losses.mean(), metrics
        finally:
            self._current_model_output = None