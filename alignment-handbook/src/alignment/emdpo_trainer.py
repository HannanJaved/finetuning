# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from .norm_dpo_trainer import NormDPOTrainer


class EMDPOTrainer(NormDPOTrainer):
    """EM-style latent reliability DPO with a structured multi-signal posterior.

    The latent reliability posterior is estimated from intrinsic pair features:
    normalized reference margin, normalized policy margin, policy/reference sign
    agreement, and a completion-length balance term. The M-step then optimizes
    DPO with the inferred posterior as per-example weights.
    """

    _name = "EMDPO"

    @staticmethod
    def _safe_metric_reduce(tensor: torch.Tensor) -> tuple[float, float, float] | None:
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.numel() == 0:
            return None
        tensor = tensor.detach()
        return tensor.mean().item(), tensor.min().item(), tensor.max().item()

    def _compute_emdpo_weights(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, dict[str, torch.FloatTensor]]:
        model_output = getattr(self, "_current_model_output", None)
        if model_output is None:
            raise ValueError("EM-DPO requires cached model_output to compute latent reliability features.")

        if getattr(self.args, "length_normalize_logps", False):
            normalized = self._apply_length_normalization(
                chosen_logps=chosen_logps,
                rejected_logps=rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                model_output=model_output,
            )
            policy_chosen, policy_rejected, ref_chosen, ref_rejected = normalized
        else:
            policy_chosen, policy_rejected = chosen_logps, rejected_logps
            ref_chosen, ref_rejected = ref_chosen_logps, ref_rejected_logps

        margin_clip = float(getattr(self.args, "emdpo_margin_clip", 5.0))
        ref_margin = torch.clamp(ref_chosen - ref_rejected, min=-margin_clip, max=margin_clip)
        policy_margin = torch.clamp(policy_chosen - policy_rejected, min=-margin_clip, max=margin_clip)

        agreement = torch.sign(ref_margin) * torch.sign(policy_margin)

        chosen_lengths = model_output["chosen_lengths"].to(policy_margin.device, dtype=policy_margin.dtype)
        rejected_lengths = model_output["rejected_lengths"].to(policy_margin.device, dtype=policy_margin.dtype)
        length_balance = -(chosen_lengths - rejected_lengths).abs() / torch.clamp(
            chosen_lengths + rejected_lengths, min=1.0
        )

        score = (
            float(getattr(self.args, "emdpo_ref_scale", 2.0)) * ref_margin
            + float(getattr(self.args, "emdpo_policy_scale", 1.0)) * policy_margin
            + float(getattr(self.args, "emdpo_agreement_scale", 0.75)) * agreement
            + float(getattr(self.args, "emdpo_length_scale", 0.25)) * length_balance
        )

        noise_eps = float(getattr(self.args, "emdpo_noise_eps", 0.1))
        noise_eps = min(max(noise_eps, 0.0), 0.499)

        posterior_clean = (1.0 - noise_eps) * torch.sigmoid(score) + noise_eps * torch.sigmoid(-score)
        posterior_clean = torch.clamp(posterior_clean, min=1e-6, max=1.0)

        min_weight = float(getattr(self.args, "emdpo_min_weight", 0.2))
        max_weight = float(getattr(self.args, "emdpo_max_weight", 1.0))
        weights = torch.clamp(posterior_clean, min=min_weight, max=max_weight)

        if getattr(self.args, "emdpo_detach_weights", True):
            weights = weights.detach()

        diagnostics = {
            "ref_margin": ref_margin.detach(),
            "policy_margin": policy_margin.detach(),
            "agreement": agreement.detach(),
            "length_balance": length_balance.detach(),
            "posterior_score": score.detach(),
        }
        return weights, diagnostics

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

            emdpo_weights = None
            emdpo_diag = None
            emdpo_enabled = bool(getattr(self.args, "emdpo_enabled", False))
            emdpo_warmup_steps = int(getattr(self.args, "emdpo_warmup_steps", 0))
            if emdpo_enabled and self.state.global_step >= emdpo_warmup_steps:
                emdpo_weights, emdpo_diag = self._compute_emdpo_weights(
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

                if emdpo_weights is not None:
                    _losses = _losses * emdpo_weights

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
            if emdpo_weights is not None and emdpo_diag is not None:
                gathered_weights = self.accelerator.gather_for_metrics(emdpo_weights)
                reduced_weights = self._safe_metric_reduce(gathered_weights)
                if reduced_weights is not None:
                    weight_mean, weight_min, weight_max = reduced_weights
                    metrics[f"{prefix}emdpo/weight_mean"] = weight_mean
                    metrics[f"{prefix}emdpo/weight_min"] = weight_min
                    metrics[f"{prefix}emdpo/weight_max"] = weight_max

                for metric_name, tensor in (
                    ("ref_margin_mean", emdpo_diag["ref_margin"]),
                    ("policy_margin_mean", emdpo_diag["policy_margin"]),
                    ("agreement_mean", emdpo_diag["agreement"]),
                    ("length_balance_mean", emdpo_diag["length_balance"]),
                    ("posterior_score_mean", emdpo_diag["posterior_score"]),
                ):
                    reduced = self._safe_metric_reduce(self.accelerator.gather_for_metrics(tensor))
                    if reduced is not None:
                        metrics[f"{prefix}emdpo/{metric_name}"] = reduced[0]

            return losses.mean(), metrics
        finally:
            self._current_model_output = None