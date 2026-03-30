# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from .norm_dpo_trainer import NormDPOTrainer


class EMDPOLiteTrainer(NormDPOTrainer):
    """EM-DPO-Lite reliability weighting using reference margin and optional policy margin.

    Includes an optional logistic-regression length bias term inspired by alpaca-eval-LC to avoid
    a manual length hyperparameter.
    """

    _name = "EMDPOLite"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._emdpo_lite_length_lr_coeffs: torch.Tensor | None = None
        self._emdpo_lite_length_lr_last_step = -1

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

    @staticmethod
    def _fit_length_lr_coeffs(
        features: torch.Tensor,
        targets: torch.Tensor,
        init_coeffs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fit a single-feature logistic regression via a few IRLS steps.

        This uses a fixed number of Newton updates (no tuning knobs) and clamps coefficients
        to keep the length bias term numerically stable.
        """

        device = features.device
        compute_dtype = torch.float32
        if init_coeffs is None:
            coeffs = torch.zeros(2, device=device, dtype=compute_dtype)
        else:
            coeffs = init_coeffs.to(device=device, dtype=compute_dtype)

        x = torch.stack([torch.ones_like(features), features], dim=-1).to(device=device, dtype=compute_dtype)
        y = targets.to(device=device, dtype=compute_dtype)
        y = torch.clamp(y, min=0.0, max=1.0)

        for _ in range(4):
            logits = (x * coeffs).sum(dim=-1)
            probs = torch.sigmoid(logits)
            weights = probs * (1.0 - probs)
            z = logits + (y - probs) / torch.clamp(weights, min=1e-6)

            w = torch.sqrt(torch.clamp(weights, min=1e-6))
            xw = x * w.unsqueeze(-1)
            zw = z * w

            xtx = xw.transpose(-1, -2) @ xw
            xtz = xw.transpose(-1, -2) @ zw
            coeffs = torch.linalg.solve(xtx + torch.eye(2, device=device, dtype=compute_dtype) * 1e-6, xtz)

        coeffs = torch.clamp(coeffs, min=-10.0, max=10.0)
        return coeffs.to(device=device, dtype=features.dtype)

    def _compute_length_lr_term(
        self,
        chosen_lengths: torch.Tensor,
        rejected_lengths: torch.Tensor,
        ref_margin: torch.Tensor,
        update_enabled: bool,
    ) -> torch.Tensor:
        length_diff = (chosen_lengths - rejected_lengths) / torch.clamp(
            chosen_lengths + rejected_lengths, min=1.0
        )
        agreement = (ref_margin > 0).to(dtype=length_diff.dtype)

        update_interval = int(getattr(self.args, "emdpo_lite_length_lr_interval", 0))
        current_step = int(getattr(self.state, "global_step", 0) or 0) if self.state is not None else 0
        should_update = update_enabled and update_interval <= 0
        if not should_update:
            if self._emdpo_lite_length_lr_coeffs is None and update_enabled:
                should_update = True
            elif self._emdpo_lite_length_lr_last_step < 0:
                should_update = update_enabled
            elif (current_step - self._emdpo_lite_length_lr_last_step) >= update_interval:
                should_update = update_enabled

        if should_update:
            with torch.no_grad():
                coeffs = self._fit_length_lr_coeffs(length_diff, agreement, self._emdpo_lite_length_lr_coeffs)
            self._emdpo_lite_length_lr_coeffs = coeffs.detach()
            self._emdpo_lite_length_lr_last_step = current_step
        else:
            if self._emdpo_lite_length_lr_coeffs is None:
                coeffs = torch.zeros(2, device=length_diff.device, dtype=length_diff.dtype)
                self._emdpo_lite_length_lr_coeffs = coeffs.detach()
            else:
                coeffs = self._emdpo_lite_length_lr_coeffs

        length_logit = coeffs[0] + coeffs[1] * length_diff
        return length_logit

    def _compute_emdpo_lite_weights(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        model_output: dict[str, torch.FloatTensor],
        update_length_lr: bool,
    ) -> tuple[torch.FloatTensor, dict[str, torch.FloatTensor]]:
        if getattr(self.args, "length_normalize_logps", False):
            policy_chosen, policy_rejected, ref_chosen, ref_rejected = self._apply_length_normalization(
                chosen_logps,
                rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                model_output,
            )
        else:
            policy_chosen, policy_rejected = chosen_logps, rejected_logps
            ref_chosen, ref_rejected = ref_chosen_logps, ref_rejected_logps

        margin_clip = float(getattr(self.args, "emdpo_lite_margin_clip", 5.0))
        ref_margin = torch.clamp(ref_chosen - ref_rejected, min=-margin_clip, max=margin_clip)

        ref_scale = float(getattr(self.args, "emdpo_lite_ref_scale", 1.0))
        policy_scale = float(getattr(self.args, "emdpo_lite_policy_scale", 0.0))
        score = ref_scale * ref_margin

        policy_margin = None
        if policy_scale != 0.0:
            policy_margin = torch.clamp(policy_chosen - policy_rejected, min=-margin_clip, max=margin_clip)
            score = score + policy_scale * policy_margin

        length_logit = None
        if bool(getattr(self.args, "emdpo_lite_length_lr_enabled", True)):
            chosen_lengths = model_output["chosen_lengths"].to(score.device, dtype=score.dtype)
            rejected_lengths = model_output["rejected_lengths"].to(score.device, dtype=score.dtype)
            length_logit = self._compute_length_lr_term(
                chosen_lengths,
                rejected_lengths,
                ref_margin,
                update_length_lr,
            )
            score = score + length_logit

        noise_eps = float(getattr(self.args, "emdpo_lite_noise_eps", 0.1))
        noise_eps = min(max(noise_eps, 0.0), 0.499)

        posterior_clean = (1.0 - noise_eps) * torch.sigmoid(score) + noise_eps * torch.sigmoid(-score)
        posterior_clean = torch.clamp(posterior_clean, min=1e-6, max=1.0)

        min_weight = float(getattr(self.args, "emdpo_lite_min_weight", 0.2))
        max_weight = float(getattr(self.args, "emdpo_lite_max_weight", 1.0))
        weights = torch.clamp(posterior_clean, min=min_weight, max=max_weight)

        if getattr(self.args, "emdpo_lite_detach_weights", True):
            weights = weights.detach()

        diagnostics = {
            "ref_margin": ref_margin.detach(),
        }
        if policy_margin is not None:
            diagnostics["policy_margin"] = policy_margin.detach()
        if length_logit is not None:
            diagnostics["length_logit"] = length_logit.detach()
            diagnostics["length_lr_intercept"] = torch.full_like(ref_margin.detach(), self._emdpo_lite_length_lr_coeffs[0])
            diagnostics["length_lr_coef"] = torch.full_like(ref_margin.detach(), self._emdpo_lite_length_lr_coeffs[1])

        return weights, diagnostics

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        if getattr(self.args, "use_liger_kernel", False):
            return super().get_batch_loss_metrics(model, batch, train_eval)

        model_output = self.concatenated_forward(model, batch)

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

        emdpo_lite_enabled = bool(getattr(self.args, "emdpo_lite_enabled", False))
        emdpo_lite_warmup_steps = int(getattr(self.args, "emdpo_lite_warmup_steps", 0))
        if not (emdpo_lite_enabled and self.state.global_step >= emdpo_lite_warmup_steps):
            return super().get_batch_loss_metrics(model, batch, train_eval)

        emdpo_weights, emdpo_diag = self._compute_emdpo_lite_weights(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            model_output,
            update_length_lr=(train_eval == "train"),
        )

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

        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )

        gathered_weights = self.accelerator.gather_for_metrics(emdpo_weights)
        reduced_weights = self._safe_metric_reduce(gathered_weights)
        if reduced_weights is not None:
            weight_mean, weight_min, weight_max = reduced_weights
            metrics[f"{prefix}emdpo_lite/weight_mean"] = weight_mean
            metrics[f"{prefix}emdpo_lite/weight_min"] = weight_min
            metrics[f"{prefix}emdpo_lite/weight_max"] = weight_max

        reduced = self._safe_metric_reduce(
            self.accelerator.gather_for_metrics(emdpo_diag["ref_margin"])
        )
        if reduced is not None:
            metrics[f"{prefix}emdpo_lite/ref_margin_mean"] = reduced[0]

        if "policy_margin" in emdpo_diag:
            reduced = self._safe_metric_reduce(
                self.accelerator.gather_for_metrics(emdpo_diag["policy_margin"])
            )
            if reduced is not None:
                metrics[f"{prefix}emdpo_lite/policy_margin_mean"] = reduced[0]

        if "length_logit" in emdpo_diag:
            for metric_name, tensor in (
                ("length_logit_mean", emdpo_diag["length_logit"]),
                ("length_lr_intercept", emdpo_diag["length_lr_intercept"]),
                ("length_lr_coef", emdpo_diag["length_lr_coef"]),
            ):
                reduced = self._safe_metric_reduce(self.accelerator.gather_for_metrics(tensor))
                if reduced is not None:
                    metrics[f"{prefix}emdpo_lite/{metric_name}"] = reduced[0]

        return losses.mean(), metrics
