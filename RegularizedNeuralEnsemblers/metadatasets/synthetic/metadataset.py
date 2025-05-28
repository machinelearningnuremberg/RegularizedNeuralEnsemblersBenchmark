# pylint: skip-file

from __future__ import annotations

import numpy as np
import torch

from ..base_metadataset import BaseMetaDataset
from .dataset_generator import DatasetGenerator


class SyntheticMetaDataset(BaseMetaDataset):
    def __init__(
        self,
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "mse",
        num_base_functions: int = 2,
        dataset_size: int = 1000,
        search_space_size: int = 100,
        simple_coefficients: bool = True,
        sample_amplitude: bool = True,
        **kwargs,
    ):
        super().__init__(
            data_dir=None,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
        )
        self.dataset_generator = DatasetGenerator(
            num_base_functions=num_base_functions,
            dataset_size=dataset_size,
            search_space_size=search_space_size,
            sample_amplitude=sample_amplitude,
            simple_coefficients=simple_coefficients,
        )

    def get_dataset_names(self) -> list[str]:
        """Fetch the dataset names present in the meta-dataset.

        Returns:
            list: List of dataset names present in the meta-dataset.
        """

        self.dataset_random_ids = np.arange(1, 100, 20)
        return [str(x) for x in self.dataset_random_ids]

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

        """
        hp_candidates = torch.FloatTensor(self.x.reshape(-1, 1))
        hp_candidates_ids = torch.FloatTensor(np.arange(hp_candidates.shape[0]))
        return hp_candidates, hp_candidates_ids

    def _get_worst_and_best_performance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch the worst and best performance for a given dataset.


        Returns:
            tuple[torch.Tensor, torch.Tensor]:
            A tuple of tensors, namely:
                - worst_performance
                - best_performance

        """

        return None, None

    def set_state(self, dataset_name: str):
        """
        Set the dataset to be used for training and evaluation.
        This method should be called before sampling.

            Args:
                dataset_name (str): Name of the dataset.

        """

        self.dataset_name = dataset_name

        random_generator_seed = int(dataset_name)
        (
            self.y,
            self.y_base,
            self.params,
            self.x,
            self.true_p,
            self.w_base,
            self.base_functions,
        ) = self.dataset_generator.get_data(random_generator_seed=random_generator_seed)

        self.base_functions = torch.FloatTensor(self.base_functions)

        self.hp_candidates, self.hp_candidates_ids = self._get_hp_candidates_and_indices()
        (
            self.worst_performance,
            self.best_performance,
        ) = self._get_worst_and_best_performance()

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

        predictions = self.get_predictions(ensembles)
        pipeline_hps = None
        metric = None
        metric_per_pipline = None
        time_per_pipeline = None

        return pipeline_hps, metric, metric_per_pipline, time_per_pipeline

    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:
        """Get predictions for the given ensembles.

        Args:
            ensembles (list[list[int]]): Ensemble configuration. Shape is [B, N].
            B = batch size (number of ensembles), N = number of pipelines per ensemble.

        Returns:
            torch.Tensor: The predictions tensor with the probability per class. Shape is [B, N, M, C].
            M = number of samples, C = number of classes.
        """
        predictions = self.base_functions[np.array(ensembles), :].unsqueeze(-1)

        return predictions
