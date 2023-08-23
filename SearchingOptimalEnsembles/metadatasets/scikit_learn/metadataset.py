from __future__ import annotations

import numpy as np
import pandas as pd
import pipeline_bench
import torch

from ..base_metadataset import BaseMetaDataset


class ScikitLearnMetaDataset(BaseMetaDataset):
    def __init__(
        self,
        data_dir: str = "/work/dlclarge2/janowski-quicktune/ask",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        split: str = "valid",
        metric_name: str = "acc",
    ):
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            split=split,
            metric_name=metric_name,
        )

        # Scikit-leran specific attributes
        self.benchmark: pipeline_bench.Benchmark
        self.feature_dim = 196
        self.task_ids = (
            pd.read_csv(
                f"{self.data_dir}/pipeline_bench_data/openml_cc18_datasets.csv",
                usecols=[0],
            )
            .values.flatten()
            .tolist()
        )

        self._initialize()

    def _get_dataset_names(self) -> list[str]:
        return (
            pd.read_csv(
                f"{self.data_dir}/pipeline_bench_data/openml_cc18_datasets.csv",
                usecols=[1],
            )
            .values.flatten()
            .tolist()
        )

    def set_dataset(self, dataset_name: str):
        # Sci-kit learn specific attributes
        task_id = self.task_ids[self.dataset_names.index(dataset_name)]
        self.benchmark = pipeline_bench.Benchmark(
            task_id=task_id, worker_dir=self.data_dir, mode="table", lazy=False
        )

        super().set_dataset(dataset_name=dataset_name)

    def _unset_dataset(self):
        del self.benchmark
        del self.dataset_name
        del self.hp_candidates
        del self.hp_candidates_ids
        del self.feature_dim

    def _get_hp_candidates_and_indices(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=protected-access
        _hp_candidates = self.benchmark._configs.compute()

        # Convert DataFrame to a numpy array and handle NaN values.
        hp_candidates = _hp_candidates.values.astype(np.float32)
        hp_candidates[np.isnan(hp_candidates)] = 0

        # Convert the numpy array to a torch tensor.
        hp_candidates = torch.from_numpy(hp_candidates)

        # Get the indices of rows that exist in the dataframe.
        hp_candidates_ids = _hp_candidates.index.values

        return hp_candidates, hp_candidates_ids

    # TODO: add time info
    def evaluate_ensembles(
        self,
        ensembles: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(ensembles)

        pipeline_hps = np.array(
            self.benchmark.get_pipeline_features(ensembles=ensembles), dtype=np.float32
        )
        pipeline_hps[np.isnan(pipeline_hps)] = 0
        pipeline_hps = torch.from_numpy(pipeline_hps)

        splits_ids = self.benchmark.get_splits(return_array=False)
        splits = self.benchmark.get_splits(return_array=True)

        y_true = np.repeat(splits[f"y_{self.split}"].reshape(1, -1), batch_size, axis=0)

        # Retieve the predictions for each pipeline in each ensemble
        y_proba = self.benchmark(
            ensembles=ensembles,
            datapoints=splits_ids[f"X_{self.split}"],
            get_probabilities=False if self.metric_name == "acc" else True,
            aggregate=False,
        )

        if self.metric_name == "acc":
            # Step 1: Reshape the predictions and the true labels to have the same shape
            predictions = np.round(y_proba.reshape(batch_size, -1, y_true.shape[1]))

            # Step 2: Compute the accuracy for each pipeline and each ensemble
            acc_per_pipeline = (predictions == y_true[:, None, :]).mean(axis=2)
            acc_per_ensemble = acc_per_pipeline.mean(axis=1)

            metric = torch.tensor(acc_per_ensemble, dtype=torch.float32)
            metric_per_pipeline = torch.tensor(acc_per_pipeline, dtype=torch.float32)

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

        # self._unset_dataset()  # TODO: FIX THIS

        return (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            metric_per_pipeline,
        )  # , time_per_pipeline
