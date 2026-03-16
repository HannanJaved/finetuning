# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

from .norm_dpo_trainer import NormDPOTrainer


class SDPOLiteTrainer(NormDPOTrainer):
    """Simplified SDPO with length-normalized DPO margin, chosen retention, and EMA reference."""

    _name = "SDPOLite"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sdpo_lite_last_ema_step = -1

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

    def _compute_sdpo_lite_terms(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        model_output: dict[str, torch.FloatTensor],
    ) -> tuple[torch.FloatTensor, dict[str, torch.Tensor]]:
        policy_chosen, policy_rejected, ref_chosen, ref_rejected = self._apply_length_normalization(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            model_output,
        )

        h_w = policy_chosen - ref_chosen
        h_l = policy_rejected - ref_rejected
        margin = h_w - h_l

        beta = float(getattr(self.args, "sdpo_lite_beta", 1.0))
        dpo_term = -F.logsigmoid(beta * margin)
        retention_alpha = self._get_sdpo_lite_retention_alpha()
        retention_term = -retention_alpha * policy_chosen
        loss = dpo_term + retention_term

        diagnostics = {
            "chosen_reward": h_w.detach(),
            "rejected_reward": h_l.detach(),
            "margin": margin.detach(),
            "dpo_term": dpo_term.detach(),
            "retention_term": retention_term.detach(),
            "retention_alpha": torch.full_like(margin.detach(), retention_alpha),
        }
        return loss, diagnostics

    def _get_sdpo_lite_retention_alpha(self) -> float:
        start_alpha = float(getattr(self.args, "sdpo_lite_retention_alpha", 0.2))
        if not bool(getattr(self.args, "sdpo_lite_decay_retention_alpha", False)):
            return start_alpha

        end_alpha = float(getattr(self.args, "sdpo_lite_retention_alpha_final", 0.05))
        progress = 0.0
        if self.state is not None:
            max_steps = int(getattr(self.state, "max_steps", 0) or 0)
            global_step = int(getattr(self.state, "global_step", 0) or 0)
            if max_steps > 0:
                progress = min(max(global_step / max_steps, 0.0), 1.0)
        return start_alpha + (end_alpha - start_alpha) * progress

    @torch.no_grad()
    def _maybe_update_ref_model_ema(self):
        if not bool(getattr(self.args, "sdpo_lite_ref_ema_enabled", True)):
            return
        if self.ref_model is None:
            return
        if self.state is not None:
            current_step = int(getattr(self.state, "global_step", 0))
            if current_step <= self._sdpo_lite_last_ema_step:
                return
            self._sdpo_lite_last_ema_step = current_step

        tau = float(getattr(self.args, "sdpo_lite_ref_ema_tau", 0.99))
        tau = min(max(tau, 0.0), 0.999999)

        for ref_param, model_param in zip(self.ref_model.parameters(), self.model.parameters()):
            ref_param.data.mul_(tau).add_(model_param.data, alpha=1.0 - tau)

        ref_buffers = dict(self.ref_model.named_buffers())
        for name, model_buffer in self.model.named_buffers():
            ref_buffer = ref_buffers.get(name)
            if ref_buffer is None or not torch.is_floating_point(ref_buffer):
                continue
            ref_buffer.data.mul_(tau).add_(model_buffer.data, alpha=1.0 - tau)

    def on_step_end(self, args, state, control, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        self._maybe_update_ref_model_ema()
        return control

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        if getattr(self.args, "use_liger_kernel", False):
            return super().get_batch_loss_metrics(model, batch, train_eval)

        metrics = {}
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

        sdpo_lite_enabled = bool(getattr(self.args, "sdpo_lite_enabled", False))
        sdpo_lite_warmup_steps = int(getattr(self.args, "sdpo_lite_warmup_steps", 0))
        if not (sdpo_lite_enabled and self.state.global_step >= sdpo_lite_warmup_steps):
            return super().get_batch_loss_metrics(model, batch, train_eval)

        losses, diag = self._compute_sdpo_lite_terms(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            model_output,
        )

        chosen_rewards = diag["chosen_reward"]
        rejected_rewards = diag["rejected_reward"]
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

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

        for metric_name, tensor in (
            ("chosen_reward_mean", diag["chosen_reward"]),
            ("rejected_reward_mean", diag["rejected_reward"]),
            ("margin_mean", diag["margin"]),
            ("dpo_term_mean", diag["dpo_term"]),
            ("retention_term_mean", diag["retention_term"]),
            ("retention_alpha", diag["retention_alpha"]),
        ):
            reduced = self._safe_metric_reduce(self.accelerator.gather_for_metrics(tensor))
            if reduced is not None:
                metrics[f"{prefix}sdpo_lite/{metric_name}"] = reduced[0]

        metrics[f"{prefix}sdpo_lite/ref_ema_tau"] = float(getattr(self.args, "sdpo_lite_ref_ema_tau", 0.99))
        return losses.mean(), metrics