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

import logging

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets

from .configs import ScriptArguments


logger = logging.getLogger(__name__)


def _maybe_create_test_split(dataset: Dataset | DatasetDict, args: ScriptArguments) -> DatasetDict:
    """Ensure a test split exists when requested via configuration."""
    if args.dataset_test_split_size is None:
        return dataset

    test_split = args.dataset_test_split
    train_split = args.dataset_train_split
    seed = args.dataset_test_split_seed

    if isinstance(dataset, Dataset):
        split = dataset.train_test_split(test_size=args.dataset_test_split_size, seed=seed)
        logger.info(
            "Created test split '%s' from dataset with test_size=%s", test_split, args.dataset_test_split_size
        )
        return DatasetDict({train_split: split["train"], test_split: split["test"]})

    if test_split in dataset:
        return dataset
    if train_split not in dataset:
        logger.warning(
            "Requested test split '%s' but train split '%s' not found. Available splits: %s",
            test_split,
            train_split,
            list(dataset.keys()),
        )
        return dataset

    split = dataset[train_split].train_test_split(test_size=args.dataset_test_split_size, seed=seed)
    logger.info(
        "Created test split '%s' from '%s' with test_size=%s",
        test_split,
        train_split,
        args.dataset_test_split_size,
    )
    return DatasetDict({**dataset, train_split: split["train"], test_split: split["test"]})


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)
        return _maybe_create_test_split(dataset, args)
    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                dataset = combined_dataset
            else:
                dataset = DatasetDict({"train": combined_dataset})

            return _maybe_create_test_split(dataset, args)
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
