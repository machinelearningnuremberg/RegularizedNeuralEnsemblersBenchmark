from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from ..base_metadataset import BaseMetaDataset


class QuicktuneMetaDataset(BaseMetaDataset):
    def __init__(
        self,
        data_dir: str,
        meta_split_ids=((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "val",
        metric_name: str = "error",
        ensemble_type: str = "soft",
        data_version: str = "micro",
    ):
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
        )
        self.ensemble_type = ensemble_type
        self.data_version = data_version

        self.dataset_name: str = ""
        self.hps = pd.read_csv(
            os.path.join(self.data_dir, data_version, "preprocessed_args.csv")
        )
        self._aggregate_info()
        self._initialize()

        self.best_performance_idx = None
        self.best_performance = None
        self.worst_performance_idx = None
        self.worst_performance = None
        self.hp_candidates: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.time: torch.Tensor = None
        self.predictions: torch.Tensor = None
        self.is_test_id: np.array | torch.Tensor = None

    def _aggregate_info(self):
        self.aggregated_info = {}
        self.dataset_names = os.listdir(os.path.join(self.data_dir, "per_dataset"))
        self.dataset_names = [
            dataset for dataset in self.dataset_names if self.data_version in dataset
        ]

        for dataset in self.dataset_names:
            # read json
            with open(
                os.path.join(self.data_dir, "per_dataset", dataset, "time_info.json")
            ) as f:
                time_info = pd.DataFrame(json.load(f))

            self.aggregated_info[dataset] = {
                "predictions": np.load(
                    os.path.join(self.data_dir, "per_dataset", dataset, "predictions.npy")
                ),
                "targets": np.load(
                    os.path.join(self.data_dir, "per_dataset", dataset, "targets.npy")
                ),
                "split_indicator": np.load(
                    os.path.join(
                        self.data_dir, "per_dataset", dataset, "split_indicator.npy"
                    )
                ),
                "time_info": time_info,
            }

    def get_statistics_dataset(self):
        # hp_candidates, time_info, predictions, targets = self.get_dataset_info(dataset_name)
        n_candidates = len(self.hp_candidates)
        pipeline_ids = np.arange(n_candidates).reshape(-1, 1).tolist()
        _, metric_per_pipeline, _, _ = self.evaluate_ensembles(pipeline_ids)
        self.best_performance = torch.max(metric_per_pipeline)
        self.best_performance_idx = torch.argmax(metric_per_pipeline)
        self.worst_performance = torch.min(metric_per_pipeline)
        self.worst_performance_idx = torch.argmin(metric_per_pipeline)

        return (
            self.best_performance,
            self.best_performance_idx,
            self.worst_performance,
            self.worst_performance_idx,
        )

    def get_dataset_info(
        self,
        dataset_name: str = "",
    ):
        self.targets = torch.LongTensor(self.aggregated_info[dataset_name]["targets"])
        self.predictions = torch.FloatTensor(
            self.aggregated_info[dataset_name]["predictions"]
        )
        self.is_test_id = np.array(self.aggregated_info[dataset_name]["split_indicator"])
        # self.time_info = self.aggregated_info[dataset_name]["time_info"]
        self.time = torch.FloatTensor(
            self.aggregated_info[dataset_name]["time_info"]["train_time"]
        )
        new_dataset_name = dataset_name.replace("_v1", "")
        self.hp_candidates = self.hps[
            self.hps[f"cat__dataset_mtlbm_{new_dataset_name}"] == 1
        ]
        self.hp_candidates = self.hp_candidates[
            [x for x in self.hps.columns if not x.startswith("cat__dataset_mtlbm")]
        ]

        self.hp_candidates = torch.FloatTensor(self.hp_candidates.values)
        self.is_test_id = torch.FloatTensor(self.is_test_id)

        if self.split == "val":
            self.predictions = self.predictions[:, self.is_test_id == 0, :]
            self.targets = self.targets[self.is_test_id == 0]


        elif self.split == "test":
            self.predictions = self.predictions[:, self.is_test_id == 1, :]
            self.targets = self.targets[self.is_test_id == 1]
        else:
            raise ValueError("split must be either val or test")

        self.predictions[torch.isnan(self.predictions)] = 0
        self.hp_candidates_ids =  torch.arange(len(self.hp_candidates))

        return self.hp_candidates, self.time, self.predictions, self.targets

    def set_dataset(self, dataset_name: str):
        if dataset_name != self.dataset_name:
            self.dataset_name = dataset_name
            self.get_dataset_info(dataset_name)
            # self.get_statistics_dataset()

    def get_dataset_names(self) -> list[str]:
        return self.dataset_names

    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hp_candidates, self.hp_candidates_ids

    def evaluate_ensembles(
        self, ensembles: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(ensembles)
        ensembles = torch.LongTensor(ensembles)
        # hp_candidates, time, predictions, targets = self.get_dataset_info(self.dataset_name)

        time_per_pipeline = self.time[ensembles]
        predictions = self.predictions[ensembles]
        targets = torch.tile(self.targets.unsqueeze(0), (batch_size, 1))
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        if batch_size == 1:
            predictions = self.predictions.unsqueeze(0)
            if len(ensembles) == 1:
                predictions = self.predictions.unsqueeze(1)

        n_classes = predictions.shape[-1]
        ensemble_size = predictions.shape[1]
        temp_targets = torch.tile(targets.unsqueeze(1), (1, ensemble_size, 1))
        hp_candidates = self.hp_candidates[ensembles]

        if self.metric_name == "error":
            metric_per_sample = torch.eq(
                predictions.reshape(-1, n_classes).argmax(-1), temp_targets.reshape(-1)
            ).float()
            metric_per_pipeline = metric_per_sample.reshape(
                batch_size, ensemble_size, -1
            ).mean(axis=2)
            metric_ensemble_per_sample = torch.eq(
                predictions.mean(1).reshape(-1, n_classes).argmax(-1), targets.reshape(-1)
            ).float()
            metric = metric_ensemble_per_sample.reshape(batch_size, -1).mean(axis=-1)

        elif self.metric_name == "nll":
            metric_per_sample = cross_entropy(
                predictions.reshape(-1, n_classes), temp_targets.reshape(-1)
            )
            metric_per_pipeline = metric_per_sample.reshape(
                batch_size, ensemble_size, -1
            ).mean(axis=-1)
            metric_ensemble_per_sample = cross_entropy(
                predictions.mean(1).reshape(-1, n_classes), targets.reshape(-1)
            )
            metric = metric_ensemble_per_sample.reshape(batch_size, -1).mean(axis=-1)

        else:
            raise ValueError("metric_name must be either acc or nll")

        return hp_candidates, metric, metric_per_pipeline, time_per_pipeline
