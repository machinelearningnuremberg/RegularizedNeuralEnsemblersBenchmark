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
    ):
        super().__init__(data_dir=data_dir, meta_split_ids=meta_split_ids)
        self.task_ids = (
            pd.read_csv(
                f"{self.data_dir}/pipeline_bench_data/openml_cc18_datasets.csv",
                usecols=[0],
            )
            .values.flatten()
            .tolist()
        )

        self._initialize()

        self.benchmark: pipeline_bench.Benchmark
        self.dataset_name: str

    def get_dataset_names(self) -> list[str]:
        return (
            pd.read_csv(
                f"{self.data_dir}/pipeline_bench_data/openml_cc18_datasets.csv",
                usecols=[1],
            )
            .values.flatten()
            .tolist()
        )

    def _set_dataset(
        self, dataset_name: str | None = None, meta_split: str = "meta-train"
    ):
        if dataset_name is None:
            self.dataset_name = np.random.choice(self.split_indices[meta_split])
        else:
            self.dataset_name = dataset_name

        task_id = self.task_ids[self.dataset_names.index(self.dataset_name)]

        self.benchmark = pipeline_bench.Benchmark(
            task_id=task_id, worker_dir=self.data_dir, mode="table", lazy=False
        )

    def _unset_dataset(self):
        del self.benchmark
        del self.dataset_name

    def get_hp_candidates(
        self, dataset_name: str, return_indices: bool = False
    ) -> torch.Tensor:
        if not hasattr(self, "benchmark"):
            self._set_dataset(dataset_name=dataset_name)

        # pylint: disable=protected-access
        hp_candidates = self.benchmark._configs.compute()
        if not return_indices:
            # Convert DataFrame to a numpy array and handle NaN values.
            hp_candidates = hp_candidates.values.astype(np.float32)
            hp_candidates[np.isnan(hp_candidates)] = 0

            # Convert the numpy array to a torch tensor.
            hp_candidates = torch.from_numpy(hp_candidates)
        else:
            # Return the indices of rows that exist in the dataframe.
            hp_candidates = hp_candidates.index.values

        return hp_candidates

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
        self._set_dataset(dataset_name=dataset_name, meta_split=meta_split)

        if observed_pipeline_ids is None:
            candidates = self.get_hp_candidates(
                dataset_name=self.dataset_name, return_indices=True
            )
        else:
            candidates = observed_pipeline_ids

        # Generate ensemble indices
        num_pipelines = np.random.randint(1, max_num_pipelines + 1)
        selected_indices = torch.multinomial(
            torch.ones(len(candidates)), num_pipelines * batch_size, replacement=True
        )

        # Fetch the actual pipeline IDs or row indices using the selected indices
        ensembles = [
            candidates[idx].tolist()
            for idx in selected_indices.view(batch_size, num_pipelines)
        ]

        (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            time_per_pipeline,
        ) = self.evaluate_ensembles(
            ensembles=ensembles,
            dataset_name=self.dataset_name,
            split=split,
            metric_name=metric_name,
        )

        return pipeline_hps, metric, metric_per_pipeline, time_per_pipeline

    # TODO: add time info
    def evaluate_ensembles(
        self,
        ensembles: list[list[int]],
        dataset_name: str,
        split: str = "valid",
        metric_name: str = "acc",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._set_dataset(dataset_name=dataset_name)

        batch_size = len(ensembles)

        pipeline_hps = np.array(
            self.benchmark.get_pipeline_features(ensembles=ensembles), dtype=np.float32
        )
        pipeline_hps[np.isnan(pipeline_hps)] = 0
        pipeline_hps = torch.from_numpy(pipeline_hps)

        splits_ids = self.benchmark.get_splits(return_array=False)
        splits = self.benchmark.get_splits(return_array=True)

        y_true = np.repeat(splits[f"y_{split}"].reshape(1, -1), batch_size, axis=0)

        # Retieve the predictions for each pipeline in each ensemble
        y_proba = self.benchmark(
            ensembles=ensembles,
            datapoints=splits_ids[f"X_{split}"],
            get_probabilities=False if metric_name == "acc" else True,
            aggregate=False,
        )

        if metric_name == "acc":
            # Step 1: Reshape the predictions and the true labels to have the same shape
            predictions = np.round(y_proba.reshape(batch_size, -1, y_true.shape[1]))

            # Step 2: Compute the accuracy for each pipeline and each ensemble
            acc_per_pipeline = (predictions == y_true[:, None, :]).mean(axis=2)
            acc_per_ensemble = acc_per_pipeline.mean(axis=1)

            metric = torch.tensor(acc_per_ensemble, dtype=torch.float32)
            metric_per_pipeline = torch.tensor(acc_per_pipeline, dtype=torch.float32)

        elif metric_name == "nll":
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

        self._unset_dataset()

        return (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            metric_per_pipeline,
        )  # , time_per_pipeline
