# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from .emdpo_trainer import EMDPOTrainer


class EMDPOWithDecayTrainer(EMDPOTrainer):
    """EM-DPO with epsilon annealing (noise-floor decay).

    Extends EMDPOTrainer with a linearly-decaying noise floor (epsilon)
    for the reliability posterior.  Every pair gets the same static eps in v1,
    but the policy signal is noisy early in training and more trustworthy late.
    Annealing eps from a high value to a low value lets the weights differentiate
    more aggressively as training progresses, without any architecture changes.

    Schedule
    --------
    ε(t) = ε_max − (ε_max − ε_min) · (t / T)

    where t = global_step and T = total training steps.

    New config fields
    -----------------
    emdpo_decay_eps_max          (float, default 0.30) – starting noise floor
    emdpo_decay_eps_min          (float, default 0.10) – final noise floor
    emdpo_decay_total_steps      (int,   default 0)    – override for total steps;
                                                          falls back to state.max_steps /
                                                          args.max_steps when 0
    emdpo_decay_disable_length_norm (bool, default True) – when True, length
                                                          normalization is disabled
                                                          inside _compute_emdpo_weights
                                                          regardless of the global
                                                          ``length_normalize_logps``
                                                          setting.  The global flag
                                                          still applies to the DPO
                                                          loss itself.

    All v1 config fields (emdpo_*) are inherited unchanged.
    The static ``emdpo_noise_eps`` field is ignored when this trainer is active
    because the schedule fully controls eps.
    """

    _name = "EMDPOWithDecay"

    def _current_noise_eps(self) -> float:
        """Return ε(t) according to the linear decay schedule.

        During evaluation (model not in training mode) we return ε_min so that
        eval metrics reflect the most-discriminative weighting.
        """
        eps_max = float(getattr(self.args, "emdpo_decay_eps_max", 0.30))
        eps_min = float(getattr(self.args, "emdpo_decay_eps_min", 0.10))

        # Clamp to valid range for the posterior formula
        eps_max = min(max(eps_max, 0.0), 0.499)
        eps_min = min(max(eps_min, 0.0), eps_max)

        if not self.model.training:
            return eps_min

        total_steps = int(getattr(self.args, "emdpo_decay_total_steps", 0))
        if total_steps <= 0:
            total_steps = int(getattr(self.state, "max_steps", 0))
        if total_steps <= 0:
            total_steps = int(getattr(self.args, "max_steps", 0))

        if total_steps <= 0:
            # Can't compute a schedule without knowing T; stay conservative.
            return eps_max

        t = int(self.state.global_step)
        # Clamp so we never go below eps_min once we reach the end.
        frac = min(t / total_steps, 1.0)
        return eps_max - (eps_max - eps_min) * frac

    def _compute_emdpo_weights(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, dict[str, torch.FloatTensor]]:
        # Temporarily patch args so the parent sees the scheduled eps and,
        # optionally, has length normalization disabled for the weight signal.
        scheduled_eps = self._current_noise_eps()
        disable_len_norm = bool(getattr(self.args, "emdpo_decay_disable_length_norm", True))

        _orig_eps = getattr(self.args, "emdpo_noise_eps", None)
        _orig_len_norm = getattr(self.args, "length_normalize_logps", None)
        try:
            self.args.emdpo_noise_eps = scheduled_eps
            if disable_len_norm:
                self.args.length_normalize_logps = False
            weights, diagnostics = super()._compute_emdpo_weights(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
        finally:
            if _orig_eps is None:
                try:
                    delattr(self.args, "emdpo_noise_eps")
                except AttributeError:
                    pass
            else:
                self.args.emdpo_noise_eps = _orig_eps

            if disable_len_norm:
                if _orig_len_norm is None:
                    try:
                        delattr(self.args, "length_normalize_logps")
                    except AttributeError:
                        pass
                else:
                    self.args.length_normalize_logps = _orig_len_norm

        # Attach the current eps to diagnostics so it is logged.
        diagnostics["noise_eps"] = torch.tensor(scheduled_eps, dtype=torch.float32)
        return weights, diagnostics

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        losses, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        # Append the scheduled epsilon to the metrics (always, not just when
        # emdpo is active, so we can verify the schedule in tensorboard even
        # before emdpo_warmup_steps is reached).
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}emdpo/noise_eps"] = self._current_noise_eps()

        return losses, metrics
