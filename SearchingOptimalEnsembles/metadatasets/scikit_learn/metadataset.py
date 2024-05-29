from __future__ import annotations

import numpy as np
import pandas as pd
import pipeline_bench
import torch

from ..base_metadataset import BaseMetaDataset
from .utils import calculate_errors


class ScikitLearnMetaDataset(BaseMetaDataset):
    def __init__(
        self,
        data_dir: str = "/work/dlclarge2/janowski-quicktune/pipeline_bench",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "nll",
        data_version: str = "mini",
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
        )

        self.feature_dim = 196

        # Scikit-learn specific attributes
        self.data_version = data_version
        self.benchmark: pipeline_bench.Benchmark
        self.task_ids = (
            pd.read_csv(
                f"{self.data_dir}/pipeline_bench_data/openml_cc18_datasets.csv",
                usecols=[0],
            )
            .values.flatten()
            .tolist()
        )

        self._initialize()

    def get_dataset_names(self) -> list[str]:
        return (
            pd.read_csv(
                f"{self.data_dir}/pipeline_bench_data/openml_cc18_datasets.csv",
                usecols=[1],
            )
            .values.flatten()
            .tolist()
        )

    def set_state(self, dataset_name: str):
        self.logger.debug(f"Setting dataset: {dataset_name}")

        # Scikit-learn specific attributes
        task_id = self.task_ids[self.dataset_names.index(dataset_name)]
        self.benchmark = pipeline_bench.Benchmark(
            task_id=task_id,
            worker_dir=self.data_dir,
            mode="table",
            lazy=False,
            data_version=self.data_version,
        )
        super().set_state(
            dataset_name=dataset_name,
        )

    def _get_hp_candidates_and_indices(
        self, return_only_ids: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hp_candidates_ids = torch.tensor(
            self.benchmark.get_hp_candidates_ids(), dtype=torch.int32
        )

        if not return_only_ids:
            # pylint: disable=protected-access
            _hp_candidates = self.benchmark._configs.compute()
            # Convert DataFrame to a numpy array and handle NaN values.
            hp_candidates = _hp_candidates.values.astype(np.float32)
            hp_candidates[np.isnan(hp_candidates)] = 0

            # Convert the numpy array to a torch tensor.
            hp_candidates = torch.from_numpy(hp_candidates)

            return hp_candidates, hp_candidates_ids
        return (
            None,
            hp_candidates_ids,
        )  # TODO: eleminate indices (future work), edit: depned only on ids

    def _get_worst_and_best_performance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.benchmark.get_worst_and_best_performance(
            split=self.split,
            metric_name=self.metric_name,
        )

    # TODO: add time info
    def evaluate_ensembles(
        self,
        ensembles: list[list[int]],
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(ensembles)
        splits = self.benchmark.get_splits(return_array=True)
        y_true = np.repeat(splits[f"y_{self.split}"].reshape(1, -1), batch_size, axis=0)
        pipeline_hps, y_probabilities = self._get_pipelines_and_probabilities(
            ensembles=ensembles
        )

        y_proba_weighted = y_probabilities.copy()
        if weights is not None:
            # since the weights have a different shape order, we need to permute the axes
            if isinstance(weights, torch.Tensor) or isinstance(weights, np.ndarray):
                weights = weights.permute(0, 2, 1, 3)
                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().detach().numpy()
            weights = weights.reshape(
                batch_size, -1, y_probabilities.shape[-2], y_probabilities.shape[-1]
            )
            y_proba_weighted *= weights

        if self.metric_name == "error":
            # Calculate error per pipeline without aggregation
            error_per_pipeline = calculate_errors(y_probabilities, y_true)
            # Aggregate the probabilities for ensemble error
            y_proba_aggregated = np.mean(y_proba_weighted, axis=-2, keepdims=True)
            error_per_ensemble = calculate_errors(y_proba_aggregated, y_true)

            metric_per_pipeline = torch.tensor(error_per_pipeline, dtype=torch.float32)
            metric = torch.tensor(error_per_ensemble, dtype=torch.float32)

        elif self.metric_name == "nll":
            raise NotImplementedError("NLL is not implemented yet")
        else:
            raise NotImplementedError

        return (
            torch.from_numpy(pipeline_hps),
            metric,
            metric_per_pipeline,
            metric_per_pipeline,
        )  # , time_per_pipeline

    def evaluate_ensembles_with_weights(
        self,
        ensembles: list[list[int]],
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.evaluate_ensembles(ensembles=ensembles, weights=weights)

    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:
        _, y_proba_np = self._get_pipelines_and_probabilities(ensembles=ensembles)
        # Convert the numpy array to torch tensor
        y_proba = torch.tensor(y_proba_np, dtype=torch.float32)
        # Assuming the current shape of y_proba is (B, M, N, C)
        # Reshape y_proba to (B, N, M, C)
        # B, D, N, C = y_proba.shape
        y_proba = y_proba.permute(0, 2, 1, 3)  # Now, shape will be (B, N, M, C)

        return y_proba

    def get_num_samples(self) -> int:
        return self.benchmark.get_splits(return_array=False)[f"X_{self.split}"].shape[0]

    def get_targets(self) -> torch.Tensor:
        splits = self.benchmark.get_splits(return_array=True)
        y_true = np.repeat(splits[f"y_{self.split}"].reshape(1, -1), 1, axis=0)
        return torch.tensor(y_true, dtype=torch.float32).squeeze()

    def get_num_classes(self) -> int:
        return len(np.unique(self.get_targets().numpy()))

    def get_num_pipelines(self) -> int:
        return len(self.hp_candidates_ids)

    def _get_pipelines_and_probabilities(
        self, ensembles: list[list[int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        pipeline_hps = self.benchmark.get_pipeline_features(ensembles=ensembles)
        pipeline_hps = pipeline_hps.astype(np.float32)

        splits_ids = self.benchmark.get_splits(return_array=False)

        # Retieve the predictions for each pipeline in each ensemble
        y_proba_np = self.benchmark(
            ensembles=ensembles,
            datapoints=splits_ids[f"X_{self.split}"],
            get_probabilities=True,
            aggregate=False,
        )

        y_proba_np = self._patch_probabilites(y_proba_np)

        return pipeline_hps, y_proba_np

    def _patch_probabilites(self, y_proba: np.ndarray) -> np.ndarray:
        # Pipeline that have NaN values in their predictions will be assigned a uniform probability
        # (treating them as if they are random guesses)
        nan_mask = np.isnan(y_proba)
        config_with_nans = nan_mask.any(axis=(1, 2, 3))
        uniform_probability = 1.0 / self.get_num_classes()
        y_proba[config_with_nans, :, :, :] = uniform_probability

        return y_proba
