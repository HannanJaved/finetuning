# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from alignment.emdpo_lite_trainer import EMDPOLiteTrainer


def test_length_lr_coeff_positive_for_positive_signal():
    torch.manual_seed(0)
    length_diff = torch.linspace(-1.0, 1.0, steps=200)
    logits = 1.5 * length_diff
    probs = torch.sigmoid(logits)
    targets = torch.bernoulli(probs)

    coeffs = EMDPOLiteTrainer._fit_length_lr_coeffs(length_diff, targets)

    assert coeffs.shape == (2,)
    assert coeffs[1].item() > 0
