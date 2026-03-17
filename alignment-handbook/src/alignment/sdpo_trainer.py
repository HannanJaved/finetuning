# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

from .emdpo_trainer import EMDPOTrainer


class SDPOTrainer(EMDPOTrainer):
    """Signal-Adaptive DPO with EM-DPO posterior weighting and EMA reference updates."""

    _name = "SDPO"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sdpo_last_ema_step = -1

    def _compute_sdpo_terms(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        model_output: dict[str, torch.FloatTensor],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, dict[str, torch.Tensor]]:
        policy_chosen, policy_rejected, ref_chosen, ref_rejected = self._apply_length_normalization(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            model_output,
        )

        h_w = policy_chosen - ref_chosen
        h_l = policy_rejected - ref_rejected

        kappa = float(getattr(self.args, "sdpo_kappa", 1.0))
        phi = torch.sigmoid(kappa * h_w)
        if getattr(self.args, "sdpo_detach_phi", True):
            phi = phi.detach()

        positive_branch = -F.logsigmoid(h_w)
        negative_branch = F.logsigmoid(h_w)
        chosen_loss = phi * positive_branch + (1.0 - phi) * negative_branch
        rejected_loss = -F.logsigmoid(-h_l)

        diagnostics = {
            "phi": phi.detach(),
            "chosen_reward": h_w.detach(),
            "rejected_reward": h_l.detach(),
            "chosen_loss": chosen_loss.detach(),
            "rejected_loss": rejected_loss.detach(),
        }
        return chosen_loss, rejected_loss, (h_w > h_l).float(), diagnostics

    @torch.no_grad()
    def _maybe_update_ref_model_ema(self):
        if not bool(getattr(self.args, "sdpo_ref_ema_enabled", True)):
            return
        if self.ref_model is None:
            return
        if self.state is not None:
            current_step = int(getattr(self.state, "global_step", 0))
            if current_step <= self._sdpo_last_ema_step:
                return
            self._sdpo_last_ema_step = current_step

        tau = float(getattr(self.args, "sdpo_ref_ema_tau", 0.99))
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

            sdpo_enabled = bool(getattr(self.args, "sdpo_enabled", False))
            sdpo_warmup_steps = int(getattr(self.args, "sdpo_warmup_steps", 0))

            if sdpo_enabled and self.state.global_step >= sdpo_warmup_steps:
                chosen_loss, rejected_loss, reward_accuracies, sdpo_diag = self._compute_sdpo_terms(
                    model_output["chosen_logps"],
                    model_output["rejected_logps"],
                    ref_chosen_logps,
                    ref_rejected_logps,
                    model_output,
                )
                emdpo_weights, emdpo_diag = self._compute_emdpo_weights(
                    model_output["chosen_logps"],
                    model_output["rejected_logps"],
                    ref_chosen_logps,
                    ref_rejected_logps,
                )
                losses = (
                    float(getattr(self.args, "sdpo_lambda_w", 0.6)) * chosen_loss
                    + float(getattr(self.args, "sdpo_lambda_r", 0.4)) * rejected_loss
                )
                losses = losses * emdpo_weights
                chosen_rewards = sdpo_diag["chosen_reward"]
                rejected_rewards = sdpo_diag["rejected_reward"]
            else:
                return super().get_batch_loss_metrics(model, batch, train_eval)

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

            for metric_name, tensor in (
                ("phi_mean", sdpo_diag["phi"]),
                ("chosen_reward_mean", sdpo_diag["chosen_reward"]),
                ("rejected_reward_mean", sdpo_diag["rejected_reward"]),
                ("chosen_loss_mean", sdpo_diag["chosen_loss"]),
                ("rejected_loss_mean", sdpo_diag["rejected_loss"]),
            ):
                reduced = self._safe_metric_reduce(self.accelerator.gather_for_metrics(tensor))
                if reduced is not None:
                    metrics[f"{prefix}sdpo/{metric_name}"] = reduced[0]

            reduced_weights = self._safe_metric_reduce(self.accelerator.gather_for_metrics(emdpo_weights))
            if reduced_weights is not None:
                weight_mean, weight_min, weight_max = reduced_weights
                metrics[f"{prefix}sdpo/posterior_weight_mean"] = weight_mean
                metrics[f"{prefix}sdpo/posterior_weight_min"] = weight_min
                metrics[f"{prefix}sdpo/posterior_weight_max"] = weight_max

            for metric_name, tensor in (
                ("ref_margin_mean", emdpo_diag["ref_margin"]),
                ("policy_margin_mean", emdpo_diag["policy_margin"]),
                ("agreement_mean", emdpo_diag["agreement"]),
                ("length_balance_mean", emdpo_diag["length_balance"]),
                ("posterior_score_mean", emdpo_diag["posterior_score"]),
            ):
                reduced = self._safe_metric_reduce(self.accelerator.gather_for_metrics(tensor))
                if reduced is not None:
                    metrics[f"{prefix}sdpo/{metric_name}"] = reduced[0]

            metrics[f"{prefix}sdpo/ref_ema_tau"] = float(getattr(self.args, "sdpo_ref_ema_tau", 0.99))
            return losses.mean(), metrics
        finally:
            self._current_model_output = None