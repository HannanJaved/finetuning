# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

import trl


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )
    dataset_test_split_size: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If provided and the dataset has no test split, create one from the train split with this fraction."
            )
        },
    )
    dataset_test_split_seed: int = field(
        default=42,
        metadata={"help": "Random seed for creating the test split from the train split."},
    )
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path or identifier for the frozen reference model used during evaluation."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    rasft_enabled: bool = field(
        default=False,
        metadata={"help": "Enable Reference-Anchored SFT (RA-SFT)."},
    )
    rasft_base_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional explicit base/reference model path for RA-SFT token novelty weights."},
    )
    rasft_entropy_normalize: bool = field(
        default=True,
        metadata={"help": "Normalize token entropy by log(vocab_size) when computing RA-SFT weights."},
    )
    rasft_weight_min: float = field(
        default=0.0,
        metadata={"help": "Lower clamp for RA-SFT per-token weights."},
    )
    rasft_weight_max: float = field(
        default=1.0,
        metadata={"help": "Upper clamp for RA-SFT per-token weights."},
    )
    rasft_max_chars_per_sample: int = field(
        default=200000,
        metadata={"help": "Drop RA-SFT examples whose combined message content exceeds this many characters."},
    )


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    length_normalize_logps: bool = field(
        default=False,
        metadata={
            "help": "If True, divide policy/reference log-probs by completion length before computing the DPO objective.",
        },
    )
    length_norm_min_tokens: float = field(
        default=1.0,
        metadata={"help": "Lower bound for the length denominator to avoid divide-by-zero."},
    )
    length_norm_exponent: float = field(
        default=1.0,
        metadata={
            "help": "Exponent applied to completion length (len**exponent) when normalizing log-probs.",
        },
    )
    emdpo_enabled: bool = field(
        default=False,
        metadata={"help": "Enable EM-style latent reliability DPO."},
    )
    emdpo_noise_eps: float = field(
        default=0.1,
        metadata={"help": "Symmetric noise prior for the EM-DPO latent reliability posterior."},
    )
    emdpo_min_weight: float = field(
        default=0.2,
        metadata={"help": "Lower bound on EM-DPO latent reliability weights."},
    )
    emdpo_max_weight: float = field(
        default=1.0,
        metadata={"help": "Upper bound on EM-DPO latent reliability weights."},
    )
    emdpo_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps before enabling EM-DPO weighting."},
    )
    emdpo_detach_weights: bool = field(
        default=True,
        metadata={"help": "Detach EM-DPO latent reliability weights from autograd."},
    )
    emdpo_ref_scale: float = field(
        default=2.0,
        metadata={"help": "Coefficient for the normalized reference margin in the EM-DPO posterior."},
    )
    emdpo_policy_scale: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the normalized policy margin in the EM-DPO posterior."},
    )
    emdpo_length_scale: float = field(
        default=0.25,
        metadata={"help": "Coefficient for the length-balance feature in the EM-DPO posterior."},
    )
    emdpo_margin_clip: float = field(
        default=5.0,
        metadata={"help": "Absolute clip value for normalized margins used in the EM-DPO posterior."},
    )
    emdpo_lite_enabled: bool = field(
        default=False,
        metadata={"help": "Enable EM-DPO-Lite weighting (reference margin with optional policy margin)."},
    )
    emdpo_lite_ref_scale: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the reference margin in EM-DPO-Lite reliability scoring."},
    )
    emdpo_lite_policy_scale: float = field(
        default=0.0,
        metadata={"help": "Coefficient for the policy margin in EM-DPO-Lite reliability scoring."},
    )
    emdpo_lite_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps before enabling EM-DPO-Lite weighting."},
    )
    emdpo_lite_noise_eps: float = field(
        default=0.1,
        metadata={"help": "Symmetric noise prior for the EM-DPO-Lite latent reliability posterior."},
    )
    emdpo_lite_min_weight: float = field(
        default=0.2,
        metadata={"help": "Lower bound on EM-DPO-Lite latent reliability weights."},
    )
    emdpo_lite_max_weight: float = field(
        default=1.0,
        metadata={"help": "Upper bound on EM-DPO-Lite latent reliability weights."},
    )
    emdpo_lite_detach_weights: bool = field(
        default=True,
        metadata={"help": "Detach EM-DPO-Lite latent reliability weights from autograd."},
    )
    emdpo_lite_margin_clip: float = field(
        default=5.0,
        metadata={"help": "Absolute clip value for margins used in the EM-DPO-Lite posterior."},
    )
    emdpo_lite_length_lr_enabled: bool = field(
        default=True,
        metadata={"help": "Fit a logistic-regression length bias term for EM-DPO-Lite (alpaca-eval-LC style)."},
    )
    emdpo_lite_length_lr_interval: int = field(
        default=50,
        metadata={
            "help": (
                "Update frequency (in steps) for the EM-DPO-Lite length LR fit; set to 0 to update every step."
            )
        },
    )
    emdpo_v2_policy_warmup_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Number of optimizer steps to fully gate off the policy contribution in EM-DPO v2. "
                "During warmup, the policy margin is zeroed before computing the reliability score."
            )
        },
    )
    emdpo_v2_policy_ramp_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Number of steps to linearly ramp the EM-DPO v2 policy contribution from 0 to 1 "
                "after warmup. Set to 0 for an immediate jump to full policy weight."
            )
        },
    )
    emdpo_v2_policy_warmup_ratio: float = field(
        default=0.05,
        metadata={
            "help": (
                "Fraction of total training steps to fully gate off the policy contribution in EM-DPO v2 "
                "when emdpo_v2_policy_warmup_steps is 0. For example, 0.05 on a 2000-step run yields ~100 steps."
            )
        },
    )
    emdpo_v2_policy_ramp_ratio: float = field(
        default=0.10,
        metadata={
            "help": (
                "Fraction of total training steps to linearly ramp the policy contribution in EM-DPO v2 "
                "when emdpo_v2_policy_ramp_steps is 0. For example, 0.10 on a 2000-step run yields ~200 steps."
            )
        },
    )
    emdpo_v2_norm_eps: float = field(
        default=1e-6,
        metadata={
            "help": (
                "Epsilon added to the batch standard deviation when z-scoring each feature in "
                "EM-DPO v2. Prevents division-by-zero for constant-valued features."
            )
        },
    )
    emdpo_decay_eps_max: float = field(
        default=0.30,
        metadata={"help": "Starting noise floor (epsilon) for the EM-DPO decay schedule."},
    )
    emdpo_decay_eps_min: float = field(
        default=0.10,
        metadata={"help": "Final noise floor (epsilon) for the EM-DPO decay schedule."},
    )
    emdpo_decay_total_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Total steps for the epsilon decay schedule. "
                "0 means infer from state.max_steps / args.max_steps."
            )
        },
    )
    emdpo_decay_disable_length_norm: bool = field(
        default=True,
        metadata={
            "help": (
                "When True, length normalization is disabled inside _compute_emdpo_weights "
                "regardless of the global length_normalize_logps setting. "
                "The global flag still applies to the DPO loss itself."
            )
        },
    )
    apdo_enabled: bool = field(
        default=False,
        metadata={"help": "Enable Adaptive Proximity DPO (APDO)."},
    )
    apdo_beta0: float = field(
        default=0.1,
        metadata={"help": "Base per-example APDO scaling coefficient."},
    )
    apdo_alpha: float = field(
        default=0.5,
        metadata={"help": "Positive offset applied to the chosen APDO reward target."},
    )
    apdo_lambda_w: float = field(
        default=0.6,
        metadata={"help": "Weight on the chosen-response APDO term."},
    )
    apdo_lambda_r: float = field(
        default=0.4,
        metadata={"help": "Weight on the rejected-response APDO term."},
    )
    apdo_gamma: float = field(
        default=0.5,
        metadata={"help": "Exponent for the APDO proximity-to-scale mapping."},
    )
    apdo_similarity_mix: float = field(
        default=0.5,
        metadata={"help": "Blend between lexical overlap and length balance in APDO pair similarity."},
    )
    apdo_min_beta: float = field(
        default=0.0,
        metadata={"help": "Lower clamp for APDO per-example scaling."},
    )
    apdo_max_beta: float = field(
        default=0.1,
        metadata={"help": "Upper clamp for APDO per-example scaling."},
    )
    apdo_ref_ema_enabled: bool = field(
        default=True,
        metadata={"help": "If True, update the frozen APDO reference model via EMA of policy weights."},
    )
    apdo_ref_ema_tau: float = field(
        default=0.99,
        metadata={"help": "EMA coefficient for APDO reference updates."},
    )
    apdo_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps before enabling APDO loss shaping."},
    )
    sdpo_enabled: bool = field(
        default=False,
        metadata={"help": "Enable Signal-Adaptive DPO (SDPO)."},
    )
    sdpo_kappa: float = field(
        default=1.0,
        metadata={"help": "Sharpness of the SDPO regime signal sigmoid applied to the chosen reward."},
    )
    sdpo_lambda_w: float = field(
        default=0.6,
        metadata={"help": "Weight on the SDPO chosen-response term."},
    )
    sdpo_lambda_r: float = field(
        default=0.4,
        metadata={"help": "Weight on the SDPO rejected-response term."},
    )
    sdpo_detach_phi: bool = field(
        default=True,
        metadata={"help": "Detach the SDPO regime signal phi from autograd for stability."},
    )
    sdpo_ref_ema_enabled: bool = field(
        default=True,
        metadata={"help": "If True, update the frozen SDPO reference model via EMA of policy weights."},
    )
    sdpo_ref_ema_tau: float = field(
        default=0.99,
        metadata={"help": "EMA coefficient for SDPO reference updates."},
    )
    sdpo_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps before enabling SDPO loss shaping."},
    )
    sdpo_lite_enabled: bool = field(
        default=False,
        metadata={"help": "Enable simplified Signal-Adaptive DPO (SDPO-Lite)."},
    )
    sdpo_lite_beta: float = field(
        default=1.0,
        metadata={"help": "Margin sharpness coefficient for SDPO-Lite."},
    )
    sdpo_lite_retention_alpha: float = field(
        default=0.2,
        metadata={"help": "Chosen-response retention coefficient for SDPO-Lite."},
    )
    sdpo_lite_decay_retention_alpha: bool = field(
        default=False,
        metadata={"help": "If True, linearly decay the SDPO-Lite retention coefficient during training."},
    )
    sdpo_lite_retention_alpha_final: float = field(
        default=0.05,
        metadata={"help": "Final SDPO-Lite retention coefficient when decay is enabled."},
    )
    sdpo_lite_ref_ema_enabled: bool = field(
        default=True,
        metadata={"help": "If True, update the frozen SDPO-Lite reference model via EMA of policy weights."},
    )
    sdpo_lite_ref_ema_tau: float = field(
        default=0.99,
        metadata={"help": "EMA coefficient for SDPO-Lite reference updates."},
    )
    sdpo_lite_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps before enabling SDPO-Lite."},
    )


@dataclass
class ORPOConfig(trl.ORPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
