from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch

from ..utils.logger import get_logger

META_SPLITS = {
    0: [(0, 1, 2), (3,), (4,)],
    1: [(1, 2, 3), (4,), (0,)],
    2: [(2, 3, 4), (0,), (1,)],
    3: [(3, 4, 0), (1,), (2,)],
    4: [(4, 0, 1), (2,), (3,)],
}


class BaseMetaDataset:
    def __init__(
        self,
        data_dir: str,
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "error",
    ):
        """Initialize the BaseMetaDataset.

        Args:
            data_dir (str): Path to the directory containing the meta-dataset.
            data_pct (tuple(float, float, float ), optional):
                Percentage of data to use for meta-train, meta-val, and meta-test.
                Defaults to (0.6, 0.2, 0.2).
            seed (int, optional): Random seed. Defaults to 42.
            split (str, optional): Dataset split name. Defaults to "valid".
            metric_name (str, optional): Name of the metric. Defaults to "error".

        Attributes:
            data_dir (str): Directory path for the dataset.
            dataset_names (list): List of dataset names present in the meta-dataset.
            split_indices (dict): Dictionary containing splits for meta-training,
                                   meta-validation, and meta-testing.
            seed (int): Random seed.
            split (str): Dataset split name.
            metric_name (str): Name of the metric.
            meta_splits (dict): Dictionary containing meta train, val, and test splits.
            meta_split_ids (tuple): Tuple containing meta train, val, and test split ids.
            logger (logging.Logger): Logger.

        """

        self.data_dir = data_dir
        self.seed = seed
        self.split = split
        self.metric_name = metric_name
        self.meta_split_ids = meta_split_ids
        self.meta_splits: dict[str, list[str]] = {}
        self.logger = get_logger(name="SEO-METADATASET", logging_level="debug")

        self.feature_dim: int = None

        # To initialize call _initialize() in the child class
        self.dataset_names: list[str] = []

        # To initialize call set_dataset(dataset_name) in the child class
        self.dataset_name: str
        self.hp_candidates: torch.Tensor
        self.hp_candidates_ids: torch.Tensor

    def _initialize(self):
        """Initialize the meta-dataset. This method should be called in the child class."""
        self.dataset_names = self.get_dataset_names()
        self.meta_splits = self._get_meta_splits()

    @abstractmethod
    def get_dataset_names(self) -> list[str]:
        """Fetch the dataset names present in the meta-dataset.

        Returns:
            list: List of dataset names present in the meta-dataset.
        """

        raise NotImplementedError

    @abstractmethod
    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch hyperparameter candidates for a given dataset.

        Args:
            dataset_name (str): Name of the dataset for which hyperparameters are required.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
            A tuple of tensors, namely:
                - hp_candidates: tensor [N, F]
                - hp_candidates_ids: tensor [N]


            where N is the number of possible pipelines evaluated for a specific dataset
            and F is the number of features per pipeline. The hp_candidates_ids tensor
            contains the pipeline ids for the corresponding hyperparameter candidates.

        Raises:
            NotImplementedError: This method should be overridden by the child class.

        """

        raise NotImplementedError

    def _get_meta_splits(self) -> dict[str, list[str]]:
        """Internal method to get meta splits for datasets.

        Args:
            data_pct (tuple(tuple(float), float, float ), optional):
                ID of the cross validation partition assigned to meta-train, meta-val and meta-test.
                The dfault assumes 5-fold cross (meta-) validation.
        Returns:
            dict[str, list[str]]: Dictionary containing meta train, val, and test splits.

        """
        rnd_gen = np.random.default_rng(self.seed)
        dataset_names = self.dataset_names.copy()
        rnd_gen.shuffle(dataset_names)

        meta_train_splits, meta_val_splits, meta_test_splits = self.meta_split_ids

        meta_splits: dict[str, list[str]] = {
            "meta-train": [],
            "meta-valid": [],
            "meta-test": [],
        }
        num_splits = len(meta_train_splits) + len(meta_test_splits) + len(meta_val_splits)
        for i, dataset in enumerate(dataset_names):
            split_id = i % num_splits
            if split_id in meta_train_splits:
                meta_splits["meta-train"].append(dataset)
            elif split_id in meta_test_splits:
                meta_splits["meta-test"].append(dataset)
            elif split_id in meta_val_splits:
                meta_splits["meta-valid"].append(dataset)
            else:
                raise ValueError("Dataset not assigned to any split")
        return meta_splits

    def set_state(self, dataset_name: str):
        """
        Set the dataset to be used for training and evaluation.
        This method should be called before sampling.

            Args:
                dataset_name (str): Name of the dataset.

        """

        self.dataset_name = dataset_name
        self.hp_candidates, self.hp_candidates_ids = self._get_hp_candidates_and_indices()

    @abstractmethod
    def evaluate_ensembles(
        self,
        ensembles: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given ensemble configurations.

        Args:
            ensembles (list[list[int]]): Ensemble configuration.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple of tensors, namely:
                - pipeline_hps: tensor [B, N, F]
                - metric: tensor [B]
                - metric_per_pipeline: tensor [B, N]
                - time_per_pipeline: tensor [B, N]

        Raises:
            NotImplementedError: This method should be overridden by the child class.

        """

        raise NotImplementedError
