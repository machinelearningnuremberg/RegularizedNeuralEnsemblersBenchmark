# type: ignore
# pylint: skip-file

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PATH = "../../AutoFinetune/aft_data/predictions/"


SPLITS = {
    0: [(0, 1, 2), (3,), (4,)],
    1: [(1, 2, 3), (4,), (0,)],
    2: [(2, 3, 4), (0,), (1,)],
    3: [(3, 4, 0), (1,), (2,)],
    4: [(4, 0, 1), (2,), (3,)],
}


class EnsembleMetaDataset:
    def __init__(
        self,
        dataset: str = None,
        batch_size: int = 16,
        max_num_pipelines: int = 10,
        split: str = "val",
        ensemble_type: str = "soft",
        split_id: int = 0,
        return_acc: bool = True,
        val_freq: int = 50,
        seed: int = 42,
        data_path: str = PATH,
        data_version: str = "micro",
        device: str = "cuda",
    ):
        self.hps = pd.read_csv(os.path.join(data_path, "preprocessed_args.csv"))
        self.batch_size = batch_size
        self.split = split
        self.max_num_pipelines = max_num_pipelines
        self.ensembling_type = ensemble_type
        self.hps_names = self.hps.columns
        self.device = device
        self.split_id = split_id
        self.return_acc = return_acc
        self.val_freq = val_freq
        self.data_path = data_path
        self.data_version = data_version
        self.observed_pipeline_ids = []
        self.random_gen = np.random.default_rng(seed)

        assert self.max_num_pipelines > 0, "max_num_models must be greater than 0"

        self._aggregate_info()

        if dataset is not None:
            self.datasets = [dataset]
        else:
            self.datasets = list(self.aggregated_info.keys())
            self._filter_datasets()
            self.N = len(self.datasets)
            self.random_gen.shuffle(self.datasets)
            train_splits, test_splits, val_splits = SPLITS[self.split_id]
            self.split_datasets = self.get_splits(
                train_splits=train_splits, test_splits=test_splits, val_splits=val_splits
            )

    def get_splits(
        self,
        datasets=None,
        seed=42,
        train_splits=(0, 1, 2),
        test_splits=(3,),
        val_splits=(4,),
    ) -> dict[str, List[str]]:
        """Splits the datasets into train, test and validation splits."""

        if datasets is None:
            datasets = self.datasets

        rnd_gen = np.random.default_rng(seed)
        rnd_gen.shuffle(datasets)

        splits = {"train": [], "val": [], "test": []}
        num_splits = len(train_splits) + len(test_splits) + len(val_splits)
        for i, dataset in enumerate(datasets):
            split_id = i % num_splits
            if split_id in train_splits:
                splits["train"].append(dataset)
            elif split_id in test_splits:
                splits["test"].append(dataset)
            elif split_id in val_splits:
                splits["val"].append(dataset)
            else:
                raise ValueError("Dataset not assigned to any split")
        return splits

    def _filter_datasets(self):
        self.datasets = [x for x in self.datasets if self.data_version in x]

    def _aggregate_info(self):
        self.aggregated_info = {}
        datasets = os.listdir(os.path.join(self.data_path, "per_dataset"))
        for dataset in datasets:
            # read json
            with open(
                os.path.join(self.data_path, "per_dataset", dataset, "time_info.json"),
            ) as f:
                time_info = pd.DataFrame(json.load(f))

            self.aggregated_info[dataset] = {
                "predictions": np.load(
                    os.path.join(
                        self.data_path, "per_dataset", dataset, "predictions.npy"
                    )
                ),
                "targets": np.load(
                    os.path.join(self.data_path, "per_dataset", dataset, "targets.npy")
                ),
                "split_indicator": np.load(
                    os.path.join(
                        self.data_path, "per_dataset", dataset, "split_indicator.npy"
                    )
                ),
                "time_info": time_info,
            }

    def set_dataset(self, dataset: str, split="val"):
        self.split = split
        self.dataset = dataset
        self.targets = np.array(self.aggregated_info[dataset]["targets"])
        self.predictions = np.array(self.aggregated_info[dataset]["predictions"])
        self.is_test_id = np.array(self.aggregated_info[dataset]["split_indicator"])
        self.time_info = self.aggregated_info[dataset]["time_info"]
        new_dataset_name = dataset.replace("_v1", "")
        self.pipeline_hps = self.hps[
            self.hps[f"cat__dataset_mtlbm_{new_dataset_name}"] == 1
        ]
        self.pipeline_hps = self.pipeline_hps[
            [x for x in self.hps.columns if not x.startswith("cat__dataset_mtlbm")]
        ]
        _, _, pipeline_perf = self.get_y_ensemble(
            np.arange(len(self.pipeline_hps)).tolist()
        )
        self.pipeline_perf = pipeline_perf.reshape(-1)
        self.best_performance = torch.max(self.pipeline_perf)
        self.best_performance_idx = torch.argmax(self.pipeline_perf)
        self.worst_performance = torch.min(self.pipeline_perf)
        self.worst_performance_idx = torch.argmin(self.pipeline_perf)

    def get_best_performance_idx(self) -> int:
        return self.best_performance_idx

    def get_worst_performance_idx(self) -> int:
        return self.worst_performance_idx

    def get_best_performance(self) -> float:
        return self.best_performance

    def get_worst_performance(self) -> float:
        return self.worst_performance

    def get_hp_candidates(self) -> pd.DataFrame:
        return self.pipeline_hps

    def get_batch(
        self,
        mode="train",
        num_encoders=1,
        max_num_pipelines=None,
        dataset=None,
        batch_size=None,
        same_input=False,
        observed_pipeline_ids=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_encoders > 1:
            multi_x = []
            multi_y = []
            multi_yp = []

            for i in range(num_encoders):
                if same_input and i > 0:
                    x, y, yp = multi_x[0], multi_y[0], multi_yp[0]
                else:
                    x, y, yp = self.get_batch(
                        mode,
                        1,
                        max_num_pipelines,
                        dataset,
                        batch_size,
                        same_input,
                        observed_pipeline_ids,
                    )
                multi_x.append(x)
                multi_y.append(y)
                multi_yp.append(yp)

            return multi_x, multi_y, multi_yp

        else:
            if max_num_pipelines is None:
                max_num_pipelines = self.max_num_pipelines

            if batch_size is None:
                batch_size = self.batch_size

            if dataset is None:
                dataset = np.random.choice(self.split_datasets[mode])

            if observed_pipeline_ids is not None:
                num_observable_pipelines = max_num_pipelines = len(observed_pipeline_ids)
                batch_size = min(batch_size, num_observable_pipelines)
            else:
                num_observable_pipelines = min(
                    max_num_pipelines, len(self.aggregated_info[dataset]["predictions"])
                )
                self.set_dataset(dataset)
            max_num_pipelines = min(max_num_pipelines, self.predictions.shape[0])
            num_pipelines = np.random.randint(1, max_num_pipelines + 1)
            pipelines_ids = np.random.randint(
                0, num_observable_pipelines, (batch_size, num_pipelines)
            )

            if observed_pipeline_ids is not None:
                pipelines_ids = np.array(observed_pipeline_ids)[pipelines_ids]

            pipeline_hps, metric, acc_per_pipeline = self.get_y_ensemble(
                pipelines_ids, batch_size
            )

            return pipeline_hps, metric, acc_per_pipeline

    def get_y_ensemble(
        self, pipelines_ids: List[int], batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time = 0
        pipelines_ids = np.array(pipelines_ids).astype(int)
        pipeline_hps = self.pipeline_hps.values[pipelines_ids]
        if self.split == "val":
            predictions = self.predictions[:, self.is_test_id == 0, :]
            targets = self.targets[self.is_test_id == 0]

        elif self.split == "test":
            predictions = self.predictions[:, self.is_test_id == 1, :]
            targets = self.targets[self.is_test_id == 1]
        else:
            raise ValueError("split must be either val or test")

        predictions[np.isnan(predictions)] = 0
        predictions = torch.FloatTensor(predictions[pipelines_ids])
        targets = torch.unsqueeze(torch.LongTensor(targets), 0)
        targets = torch.tile(targets, (batch_size, 1))
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        # TODO: double-check this implementation
        if batch_size == 1:
            predictions = predictions.unsqueeze(0)
            if len(pipelines_ids) == 1:
                predictions = predictions.unsqueeze(1)

        acc_per_pipeline = (
            (predictions.argmax(axis=-1) == targets.unsqueeze(1)).float().mean(axis=-1)
        )
        predictions = torch.mean(predictions, axis=1)
        n_classes = predictions.shape[-1]
        ce_per_sample = cross_entropy(
            predictions.reshape(-1, n_classes), targets.reshape(-1)
        )
        ce = ce_per_sample.reshape(batch_size, -1).mean(axis=1)
        acc_per_sample = (
            predictions.reshape(-1, n_classes).argmax(axis=1) == targets.reshape(-1)
        ).float()
        acc = acc_per_sample.reshape(batch_size, -1).mean(axis=1)

        pipeline_hps = torch.FloatTensor(pipeline_hps).to(self.device)
        ce = ce.to(self.device)
        acc = acc.to(self.device)

        if self.return_acc:
            metric = acc
        else:
            metric = ce

        return pipeline_hps, metric, acc_per_pipeline
