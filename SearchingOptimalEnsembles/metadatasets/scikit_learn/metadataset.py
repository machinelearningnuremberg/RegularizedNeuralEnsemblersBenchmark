from __future__ import annotations

import numpy as np
import pandas as pd
import pipeline_bench
import torch

from ..base_metadataset import BaseMetaDataset


class ScikitLearnMetaDataset(BaseMetaDataset):
    def __init__(
        self,
        data_dir: str = "/work/dlclarge2/janowski-quicktune/pipeline_bench",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "nll",
        data_version: str = "mini",
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(ensembles)

        pipeline_hps = self.benchmark.get_pipeline_features(ensembles=ensembles)
        pipeline_hps = pipeline_hps.astype(np.float32)
        pipeline_hps = torch.from_numpy(pipeline_hps)

        splits_ids = self.benchmark.get_splits(return_array=False)
        splits = self.benchmark.get_splits(return_array=True)

        y_true = np.repeat(splits[f"y_{self.split}"].reshape(1, -1), batch_size, axis=0)

        # Retieve the predictions for each pipeline in each ensemble
        y_proba = self.benchmark(
            ensembles=ensembles,
            datapoints=splits_ids[f"X_{self.split}"],
            get_probabilities=False if self.metric_name == "error" else True,
            aggregate=False,
        )

        if self.metric_name == "error":
            # Step 1: Reshape the predictions and the true labels to have the same shape
            predictions = np.round(y_proba.reshape(batch_size, -1, y_true.shape[1]))

            # Step 2: Compute the accuracy for each pipeline and each ensemble
            acc_per_pipeline = (predictions == y_true[:, None, :]).mean(axis=2)
            acc_per_ensemble = acc_per_pipeline.mean(axis=1)

            # Step 3: Compute the error rate
            error_per_pipeline = 1.0 - acc_per_pipeline
            error_per_ensemble = 1.0 - acc_per_ensemble

            metric = torch.tensor(error_per_ensemble, dtype=torch.float32)
            metric_per_pipeline = torch.tensor(error_per_pipeline, dtype=torch.float32)

        elif self.metric_name == "nll":
            num_datapoints = y_proba.shape[1]
            num_pipelines = y_proba.shape[2]

            # Step 1: Get the probabilities of the true classes
            batch_indices = np.arange(batch_size)[:, np.newaxis, np.newaxis]
            datapoint_indices = np.arange(num_datapoints)[:, np.newaxis]
            pipeline_indices = np.arange(num_pipelines)

            true_class_indices = y_true[:, :, np.newaxis]

            true_probabilities = y_proba[
                batch_indices, datapoint_indices, pipeline_indices, true_class_indices
            ]

            # Step 2: Compute the negative log of those probabilities (with a small epsilon to avoid NaNs)
            nll_per_pipeline_datapoint = -np.log(true_probabilities + 1e-10)

            # Step 3: Average over datapoints to get NLL for each pipeline
            nll_per_pipeline = nll_per_pipeline_datapoint.mean(axis=1)

            # Step 4: Average over pipelines to get a single NLL value for each ensemble
            nll_per_ensemble = nll_per_pipeline.mean(axis=1)

            metric = torch.tensor(nll_per_ensemble, dtype=torch.float32)
            metric_per_pipeline = torch.tensor(nll_per_pipeline, dtype=torch.float32)

        else:
            raise NotImplementedError

        return (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            metric_per_pipeline,
        )  # , time_per_pipeline

    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:
        pipeline_hps = self.benchmark.get_pipeline_features(ensembles=ensembles)
        pipeline_hps = pipeline_hps.astype(np.float32)
        pipeline_hps = torch.from_numpy(pipeline_hps)

        splits_ids = self.benchmark.get_splits(return_array=False)

        # Retieve the predictions for each pipeline in each ensemble
        y_proba_np = self.benchmark(
            ensembles=ensembles,
            datapoints=splits_ids[f"X_{self.split}"],
            get_probabilities=True,
            aggregate=False,
        )

        # Convert the numpy array to torch tensor
        y_proba = torch.from_numpy(y_proba_np)

        # Assuming the current shape of y_proba is (B, M, N, C)
        # Reshape y_proba to (B, N, M, C)
        # B, D, N, C = y_proba.shape
        y_proba = y_proba.permute(0, 2, 1, 3)  # Now, shape will be (B, N, M, C)

        return y_proba
