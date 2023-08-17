from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
from sklearn.model_selection import train_test_split


class BaseMetaDataset:
    def __init__(
        self,
        data_dir: str,
        data_pct: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ):
        """Initialize the BaseMetaDataset.

        Args:
            data_dir (str): Path to the directory containing the meta-dataset.
            data_pct (tuple(float, float, float ), optional):
                Percentage of data to use for meta-train, meta-val, and meta-test.
                Defaults to (0.6, 0.2, 0.2).

        Attributes:
            data_dir (str): Directory path for the dataset.
            dataset_names (list): List of dataset names present in the meta-dataset.
            split_indices (dict): Dictionary containing splits for meta-training,
                                   meta-validation, and meta-testing.

        """

        self.data_dir = data_dir
        self.dataset_names: list[str] = []
        self.split_indices = self._get_meta_splits(data_pct=data_pct)

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

    def _get_meta_splits(
        self, data_pct: tuple[float, float, float] = (0.6, 0.2, 0.2)
    ) -> dict[str, list[str]]:
        """Internal method to get meta splits for datasets.

        Args:
            data_pct (tuple(float, float, float ), optional):
                Percentage split for meta-train, meta-val, and meta-test.
                Defaults to (0.6, 0.2, 0.2).

        Returns:
            dict[str, list[str]]: Dictionary containing meta train, val, and test splits.

        """

        # Split into train and remaining meta datasets
        train, rem = train_test_split(
            np.array(self.dataset_names),
            test_size=data_pct[1] + data_pct[2],
            random_state=42,
        )

        # Split remaining meta datasets into validation and test
        val, test = train_test_split(
            rem, test_size=data_pct[2] / (data_pct[1] + data_pct[2]), random_state=42
        )

        return {"meta-train": train, "meta-val": val, "meta-test": test}

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
        self,
        ensembles: list[list[int]],
        dataset_name: str,
        split: str = "valid",
        metric_name: str = "acc",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given ensemble configurations.

        Args:
            ensembles (list[list[int]]): Ensemble configuration.
            dataset_name (str): Name of the dataset.
            split (str, optional): Dataset split name. Defaults to "valid".
            metric_name (str, optional): Name of the metric. Defaults to "acc".

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
