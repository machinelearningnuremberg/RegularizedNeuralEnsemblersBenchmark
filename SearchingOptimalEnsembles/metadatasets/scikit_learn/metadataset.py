from __future__ import annotations

from joblib import load  
import numpy as np
import pandas as pd
import pipeline_bench
import torch
from sklearn.pipeline import Pipeline

from ..evaluator import Evaluator


class ScikitLearnMetaDataset(Evaluator):
    metadataset_name = "scikit-learn"

    def __init__(
        self,
        data_dir: str = "/work/dlclarge1/janowski-pipebench/",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "nll",
        data_version: str = "mini",
        task_type: str = "classification",
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
            data_version=data_version,
        )

        self.feature_dim = 196
        self.task_type = "classification"

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

    def set_state(self, dataset_name: str, split: str = "valid"):
        self.logger.debug(f"Setting dataset: {dataset_name}")

        if dataset_name != self.dataset_name:
            # Scikit-learn specific attributes
            task_id = self.task_ids[self.dataset_names.index(dataset_name)]
            self.benchmark = pipeline_bench.Benchmark(
                task_id=task_id,
                worker_dir=self.data_dir,
                mode="table",
                lazy=False,
                data_version=self.data_version,
            )
        
        super().set_state(dataset_name=dataset_name, split=split)

    def _get_hp_candidates_and_indices(
        self, return_only_ids: bool = False
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hp_candidates_ids = torch.tensor(
            self.benchmark.get_hp_candidates_ids(), dtype=torch.int32
        )

        if not return_only_ids:
            # pylint: disable=protected-access
            _hp_candidates = self.benchmark._configs #.compute()
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
        try:
            return self.benchmark.get_worst_and_best_performance(
                split=self.split,
                metric_name=self.metric_name,
            )
        except KeyError:
            # calculate the worst and best performance
            ensembles = [[hp_id.item()] for hp_id in self.hp_candidates_ids]
            metric_per_pipeline = self.benchmark(
                ensembles=ensembles,
                datapoints=self.benchmark.get_splits(return_array=False)[
                    f"X_{self.split}"
                ],
                get_probabilities=False,
                aggregate=True,
            )
            labels = self.benchmark.get_splits(return_array=True)[f"y_{self.split}"]
            labels = np.repeat(labels.reshape(1, -1), len(ensembles), axis=0)
            labels = torch.tensor(labels, dtype=torch.long)
            # compute accruacy for each pipeline
            return (
                torch.tensor(metric_per_pipeline.min(), dtype=torch.float32),
                torch.tensor(metric_per_pipeline.max(), dtype=torch.float32),
            )

    def _get_probabilities(self, ensembles: list[list[int]]) -> np.ndarray:
        splits_ids = self.benchmark.get_splits(return_array=False)

        # Retieve the predictions for each pipeline in each ensemble
        y_proba = self.benchmark(
            ensembles=ensembles,
            datapoints=splits_ids[f"X_{self.split}"],
            get_probabilities=True,
            aggregate=False,
        )

        # Pipeline that have NaN values in their predictions will be assigned a uniform probability
        # (treating them as if they are random guesses)
        nan_mask = np.isnan(y_proba)
        y_proba[nan_mask] = 1e-4
        y_proba= y_proba / y_proba.sum(-1, keepdims=True)
        
        return y_proba

    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:
        y_proba_np = self._get_probabilities(ensembles=ensembles)
        # Convert the numpy array to torch tensor
        y_proba = torch.tensor(y_proba_np, dtype=torch.float32)
        # Assuming the current shape of y_proba is (B, M, N, C)
        y_proba = y_proba.permute(0, 2, 1, 3)  # Now, shape will be (B, N, M, C)

        return y_proba

    def get_num_samples(self) -> int:
        return self.benchmark.get_splits(return_array=False)[f"X_{self.split}"].shape[0]

    def get_targets(self) -> torch.Tensor:
        splits = self.benchmark.get_splits(return_array=True)
        y_true = np.repeat(splits[f"y_{self.split}"].reshape(1, -1), 1, axis=0)
        return torch.tensor(y_true, dtype=torch.long).squeeze()

    def get_num_classes(self) -> int:
        return len(np.unique(self.get_targets().numpy()))

    def get_num_pipelines(self) -> int:
        return len(self.hp_candidates_ids)

    def get_time(self, ensembles: list[list[int]]) -> torch.Tensor:
        return torch.zeros(len(ensembles), len(ensembles[0]))

    def get_base_pipelines(self):
        return self.get_pipelines([self.hp_candidates_ids.numpy().tolist()])

    def get_pipelines(self, ensembles: list[list[int]]) -> list[list[Pipeline]]:
        pipeline_ids = [pipeline_id for sublist in ensembles for pipeline_id in sublist]
        pipeline_paths_df = self.benchmark._pipelines[
            self.benchmark._pipelines["pipeline_id"].isin(pipeline_ids)
        ]
        id_to_path = dict(
            zip(pipeline_paths_df["pipeline_id"], pipeline_paths_df["pipeline_path"])
        )

        ens = []
        for sublist in ensembles:
            for pipeline_id in sublist:
                try:
                    with open(id_to_path[pipeline_id], 'rb') as f:
                        p = load(f)
                except Exception as e:
                    print(f"Error loading pipeline {pipeline_id}: {e}")
                    # p = Pipeline()
                    raise e
                ens.append(p)

        return ens

    def get_features(self, ensembles: list[list[int]]) -> torch.Tensor:
        pipeline_hps = self.benchmark.get_pipeline_features(ensembles=ensembles)
        pipeline_hps = pipeline_hps.astype(np.float32)
        return torch.from_numpy(pipeline_hps)

    def get_X_and_y(self) -> tuple[torch.Tensor, torch.Tensor]:
        splits = self.benchmark.get_splits(return_array=True)
        X = torch.tensor(splits[f"X_{self.split}"], dtype=torch.float32)
        y = torch.tensor(splits[f"y_{self.split}"], dtype=torch.long)
        X[torch.isnan(X)] = 0
        return X, y
