from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch

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
        metric_name: str = "acc",
    ):
        """Initialize the BaseMetaDataset.

        Args:
            data_dir (str): Path to the directory containing the meta-dataset.
            data_pct (tuple(float, float, float ), optional):
                Percentage of data to use for meta-train, meta-val, and meta-test.
                Defaults to (0.6, 0.2, 0.2).
            seed (int, optional): Random seed. Defaults to 42.

        Attributes:
            data_dir (str): Directory path for the dataset.
            dataset_names (list): List of dataset names present in the meta-dataset.
            split_indices (dict): Dictionary containing splits for meta-training,
                                   meta-validation, and meta-testing.

        """

        self.data_dir = data_dir
        self.seed = seed
        self.meta_split_ids = meta_split_ids
        # self.dataset_name = dataset_name
        self.split = split
        self.metric_name = metric_name
        # self.device = device

        # To initialize call _initialize() in the child class
        self.dataset_names: list[str] = []
        self.split_indices: dict[str, list[str]] = {}

    def _initialize(self):
        self.dataset_names = self.get_dataset_names()
        self.split_indices = self._get_meta_splits()

    def get_dataset_names(self) -> list[str]:
        """Fetch the dataset names present in the meta-dataset.

        Returns:
            list: List of dataset names present in the meta-dataset.
        """

        raise NotImplementedError

    @abstractmethod
    def get_hp_candidates(self, dataset_name: str) -> torch.Tensor:
        """Fetch hyperparameter candidates for a given dataset.

        Args:
            dataset_name (str): Name of the dataset for which hyperparameters are required.

        Returns:
            torch.Tensor: A tensor of size [N, F], where N is the number of possible
            pipelines evaluated for a specific dataset and F is the number of features
            per pipeline.

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
            "meta-val": [],
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
                meta_splits["meta-val"].append(dataset)
            else:
                raise ValueError("Dataset not assigned to any split")
        return meta_splits

    @abstractmethod
    def get_batch(
        self,
        meta_split: str = "meta-train",
        split: str = "valid",
        dataset_name: str | None = None,
        max_num_pipelines: int = 10,
        batch_size: int = 16,
        metric_name: str = "acc",
        observed_pipeline_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch a batch of data.

        Args:
            meta_split (str, optional): Meta dataset split name. Defaults to "meta-train".
            split (str, optional): Dataset split name. Defaults to "valid".
            dataset_name (str, optional): Name of the dataset. Defaults to None.
            max_num_pipelines (int, optional): Maximum number of pipelines. Defaults to 10.
            batch_size (int, optional): Size of the batch to fetch. Defaults to 16.
            metric_name (str, optional): Name of the metric. Defaults to "acc".
            observed_pipeline_ids (list[int], optional): List of observed pipeline IDs. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple of tensors, namely:
                - pipeline_hps: tensor [B, N, F]
                - metric: tensor [B]
                - metric_per_pipeline: tensor [B, N]
                - time_per_pipeline: tensor [B, N]

            where:
                - B is the batch size
                - N = Number of pipelines per ensemble, with 1 <= N <= max_num_pipelines
                - F = Number of features per pipeline

        Raises:
            NotImplementedError: This method should be overridden by the child class.

        """

        raise NotImplementedError

    @abstractmethod
    def evaluate_ensembles(
        self, ensembles: list[list[int]]
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
