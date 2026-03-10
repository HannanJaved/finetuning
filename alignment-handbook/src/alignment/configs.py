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
    lndpo_enabled: bool = field(
        default=False,
        metadata={"help": "Enable Latent Noise-Aware DPO (LN-DPO) reliability weighting on fixed preference pairs."},
    )
    lndpo_noise_eps: float = field(
        default=0.1,
        metadata={
            "help": "Symmetric preference-label noise rate used to shrink high-confidence updates in LN-DPO.",
        },
    )
    lndpo_beta_scale: float = field(
        default=1.0,
        metadata={"help": "Multiplier applied to the reference margin inside the LN-DPO reliability estimator."},
    )
    lndpo_min_weight: float = field(
        default=0.2,
        metadata={"help": "Lower bound on per-example LN-DPO weights to avoid dropping all gradient signal."},
    )
    lndpo_max_weight: float = field(
        default=1.0,
        metadata={"help": "Upper bound on per-example LN-DPO weights."},
    )
    lndpo_detach_weights: bool = field(
        default=True,
        metadata={"help": "Detach LN-DPO reliability weights from autograd for stable optimization."},
    )
    lndpo_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps before enabling LN-DPO weighting; 0 enables it immediately."},
    )
    lndpo_v2_enabled: bool = field(
        default=False,
        metadata={"help": "Enable LN-DPO v2 with bounded, length-normalized reliability weighting."},
    )
    lndpo_v2_margin_scale: float = field(
        default=2.0,
        metadata={"help": "Scale applied to the bounded normalized reference margin in LN-DPO v2."},
    )
    lndpo_v2_margin_clip: float = field(
        default=5.0,
        metadata={"help": "Absolute clip value for the bounded normalized reference margin in LN-DPO v2."},
    )
    lndpo_v2r_enabled: bool = field(
        default=False,
        metadata={"help": "Enable LN-DPO v2 with chosen-response retention (implemented via rpo_alpha)."},
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
    emdpo_agreement_scale: float = field(
        default=0.75,
        metadata={"help": "Coefficient for policy/reference sign agreement in the EM-DPO posterior."},
    )
    emdpo_length_scale: float = field(
        default=0.25,
        metadata={"help": "Coefficient for the length-balance feature in the EM-DPO posterior."},
    )
    emdpo_margin_clip: float = field(
        default=5.0,
        metadata={"help": "Absolute clip value for normalized margins used in the EM-DPO posterior."},
    )
    sgdpo_enabled: bool = field(
        default=False,
        metadata={"help": "Enable Self-Guided DPO (SG-DPO)."},
    )
    sgdpo_beta: float = field(
        default=1.0,
        metadata={"help": "Sharpness of the SG-DPO weighting sigmoid applied to the implicit reward margin."},
    )
    sgdpo_ema_alpha: float = field(
        default=1.0,
        metadata={"help": "EMA coefficient for the SG-DPO implicit reward margin; 1.0 disables smoothing."},
    )
    sgdpo_min_weight: float = field(
        default=0.2,
        metadata={"help": "Lower bound on SG-DPO per-example weights."},
    )
    sgdpo_max_weight: float = field(
        default=0.9,
        metadata={"help": "Upper bound on SG-DPO per-example weights."},
    )
    sgdpo_detach_weights: bool = field(
        default=True,
        metadata={"help": "Detach SG-DPO per-example weights from autograd."},
    )
    sgdpo_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of optimizer steps before enabling SG-DPO weighting."},
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


@dataclass
class ORPOConfig(trl.ORPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
