# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections import Counter

import torch
import torch.nn.functional as F

from .norm_dpo_trainer import NormDPOTrainer


class APDOTrainer(NormDPOTrainer):
    """Adaptive Proximity Direct Preference Optimization.

    This APDO implementation focuses on the most actionable parts of the proposal:
    decoupled chosen/rejected objectives, length-normalized implicit rewards, an
    adaptive per-example scale based on pair proximity, and an EMA-updated reference.
    """

    _name = "APDO"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apdo_last_ema_step = -1

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
    def _flatten_text(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            chunks = []
            for item in value:
                if isinstance(item, dict):
                    chunks.append(str(item.get("content", "")))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks)
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _pair_similarity(self, batch: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        chosens = batch.get("chosen", [])
        rejecteds = batch.get("rejected", [])
        mix = float(getattr(self.args, "apdo_similarity_mix", 0.5))
        mix = min(max(mix, 0.0), 1.0)

        similarities: list[float] = []
        for chosen, rejected in zip(chosens, rejecteds):
            chosen_text = self._flatten_text(chosen)
            rejected_text = self._flatten_text(rejected)

            chosen_tokens = self._tokenize(chosen_text)
            rejected_tokens = self._tokenize(rejected_text)

            if chosen_tokens or rejected_tokens:
                chosen_counts = Counter(chosen_tokens)
                rejected_counts = Counter(rejected_tokens)
                overlap = sum((chosen_counts & rejected_counts).values())
                union = sum((chosen_counts | rejected_counts).values())
                lexical_sim = overlap / max(union, 1)
            else:
                lexical_sim = 1.0

            chosen_len = max(len(chosen_tokens), 1)
            rejected_len = max(len(rejected_tokens), 1)
            length_ratio = min(chosen_len, rejected_len) / max(chosen_len, rejected_len)

            sim = mix * lexical_sim + (1.0 - mix) * length_ratio
            similarities.append(float(min(max(sim, 0.0), 1.0)))

        return torch.tensor(similarities, device=device, dtype=dtype)

    def _compute_apdo_terms(
        self,
        batch: dict,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        model_output: dict[str, torch.FloatTensor],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, dict[str, torch.Tensor]]:
        policy_chosen, policy_rejected, ref_chosen, ref_rejected = self._apply_length_normalization(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            model_output,
        )

        chosen_reward = policy_chosen - ref_chosen
        rejected_reward = policy_rejected - ref_rejected

        similarity = self._pair_similarity(batch, chosen_reward.device, chosen_reward.dtype)
        gamma = float(getattr(self.args, "apdo_gamma", 0.5))
        beta0 = float(getattr(self.args, "apdo_beta0", 0.1))
        beta_i = beta0 * torch.pow(torch.clamp(1.0 - similarity, min=0.0, max=1.0), gamma)
        beta_i = torch.clamp(
            beta_i,
            min=float(getattr(self.args, "apdo_min_beta", 0.0)),
            max=float(getattr(self.args, "apdo_max_beta", beta0)),
        )

        alpha = float(getattr(self.args, "apdo_alpha", 0.5))
        chosen_loss = -F.logsigmoid(beta_i * (chosen_reward + alpha))
        rejected_loss = -F.logsigmoid(beta_i * (-rejected_reward))

        diagnostics = {
            "beta": beta_i.detach(),
            "similarity": similarity.detach(),
            "chosen_reward": chosen_reward.detach(),
            "rejected_reward": rejected_reward.detach(),
            "chosen_loss": chosen_loss.detach(),
            "rejected_loss": rejected_loss.detach(),
        }
        return chosen_loss, rejected_loss, diagnostics

    @torch.no_grad()
    def _maybe_update_ref_model_ema(self):
        if not bool(getattr(self.args, "apdo_ref_ema_enabled", True)):
            return
        if self.ref_model is None:
            return
        if self.state is not None:
            current_step = int(getattr(self.state, "global_step", 0))
            if current_step <= self._apdo_last_ema_step:
                return
            self._apdo_last_ema_step = current_step

        tau = float(getattr(self.args, "apdo_ref_ema_tau", 0.99))
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

        apdo_enabled = bool(getattr(self.args, "apdo_enabled", False))
        apdo_warmup_steps = int(getattr(self.args, "apdo_warmup_steps", 0))

        if apdo_enabled and self.state.global_step >= apdo_warmup_steps:
            chosen_loss, rejected_loss, apdo_diag = self._compute_apdo_terms(
                batch,
                model_output["chosen_logps"],
                model_output["rejected_logps"],
                ref_chosen_logps,
                ref_rejected_logps,
                model_output,
            )
            losses = (
                float(getattr(self.args, "apdo_lambda_w", 0.6)) * chosen_loss
                + float(getattr(self.args, "apdo_lambda_r", 0.4)) * rejected_loss
            )
            chosen_rewards = apdo_diag["chosen_reward"]
            rejected_rewards = -apdo_diag["rejected_reward"]
        else:
            losses = 0
            chosen_rewards = 0
            rejected_rewards = 0
            apdo_diag = None
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

        if apdo_diag is not None:
            for metric_name, tensor in (
                ("beta_mean", apdo_diag["beta"]),
                ("similarity_mean", apdo_diag["similarity"]),
                ("chosen_reward_mean", apdo_diag["chosen_reward"]),
                ("rejected_reward_mean", apdo_diag["rejected_reward"]),
                ("chosen_loss_mean", apdo_diag["chosen_loss"]),
                ("rejected_loss_mean", apdo_diag["rejected_loss"]),
            ):
                reduced = self._safe_metric_reduce(self.accelerator.gather_for_metrics(tensor))
                if reduced is not None:
                    metrics[f"{prefix}apdo/{metric_name}"] = reduced[0]

            reduced_beta = self._safe_metric_reduce(self.accelerator.gather_for_metrics(apdo_diag["beta"]))
            if reduced_beta is not None:
                _, beta_min, beta_max = reduced_beta
                metrics[f"{prefix}apdo/beta_min"] = beta_min
                metrics[f"{prefix}apdo/beta_max"] = beta_max

            metrics[f"{prefix}apdo/ref_ema_tau"] = float(getattr(self.args, "apdo_ref_ema_tau", 0.99))

        return losses.mean(), metrics