from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from ..base_metadataset import BaseMetaDataset
from ..evaluator import Evaluator

#TODO: Unify the name "hp_candidate" and "pipeline", it might be confusing

class QuicktuneMetaDataset(Evaluator):
    
    metadataset_name = "quicktune"

    def __init__(
        self,
        data_dir: str = "/work/dlclarge2/janowski-quicktune/predictions",
        meta_split_ids=((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "error",
        ensemble_type: str = "soft",
        data_version: str = "micro",
        use_logits: bool = True,
        device: torch.device = torch.device("cpu"),
        processing_batch_size: int = 1000,
        **kwargs,
    ):
        
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
            data_version=data_version
        )

        self.feature_dim = 65

        self.ensemble_type = ensemble_type
        self.data_version = data_version
        self.use_logits = use_logits
        self.device = device
        self.processing_batch_size = processing_batch_size

        self.dataset_name: str = ""
        self.hps = pd.read_csv(
            os.path.join(self.data_dir, data_version, "preprocessed_args.csv")
        )
        self._initialize()
        self._aggregate_info()

        self.best_performance_idx = None
        self.best_performance = None
        self.worst_performance_idx = None
        self.worst_performance = None
        self.hp_candidates_ids: torch.Tensor = None
        self.hp_candidates: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.time: torch.Tensor = None
        self.predictions: torch.Tensor = None
        self.is_test_id: np.array | torch.Tensor = None

    def _aggregate_info(self, dataset_name: str = None):
        self.aggregated_info = {}

        if dataset_name is None:
            dataset_names = self.dataset_names
        else:
            dataset_names = [dataset_name]

        for dataset in dataset_names:
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
        self.best_performance = torch.min(metric_per_pipeline)
        self.best_performance_idx = torch.argmin(metric_per_pipeline)
        self.worst_performance = torch.max(metric_per_pipeline)
        self.worst_performance_idx = torch.argmax(metric_per_pipeline)
        self.metric_per_pipeline = metric_per_pipeline
        return (
            self.best_performance,
            self.best_performance_idx,
            self.worst_performance,
            self.worst_performance_idx,
        )

    def _get_worst_and_best_performance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.worst_performance, self.best_performance
    
    def get_dataset_info(
        self,
        dataset_name: str = "",
    ):
        self.is_test_id = np.array(self.aggregated_info[dataset_name]["split_indicator"])
        # self.time_info = self.aggregated_info[dataset_name]["time_info"]

        new_dataset_name = dataset_name.replace("_v1", "")
        self.hp_candidates = self.hps[
            self.hps[f"cat__dataset_mtlbm_{new_dataset_name}"] == 1
        ]
        self.hp_candidates = self.hp_candidates[
            [x for x in self.hps.columns if not x.startswith("cat__dataset_mtlbm")]
        ]

        self.hp_candidates = torch.FloatTensor(self.hp_candidates.values)
        self.hp_candidates_ids = torch.arange(len(self.hp_candidates))

        self.is_test_id = torch.FloatTensor(self.is_test_id)

        self.targets = torch.LongTensor(self.aggregated_info[dataset_name]["targets"])
        self.predictions = torch.FloatTensor(
            self.aggregated_info[dataset_name]["predictions"]
        )

        try:
            self.time = torch.FloatTensor(
                self.aggregated_info[dataset_name]["time_info"]["train_time"]
            )
        except:
            self.time = torch.zeros(len(self.hp_candidates))
            
        if self.split == "valid":
            self.targets = self.targets[self.is_test_id == 0]
            self.predictions = self.predictions[:, self.is_test_id == 0, :]

        elif self.split == "test":
            self.targets = self.targets[self.is_test_id == 1]
            self.predictions = self.predictions[:, self.is_test_id == 1, :]

        else:
            raise ValueError("split must be either val or test")

        for i in range(0, self.predictions.shape[-2], self.processing_batch_size):
            range_idx = list(
                range(i, min(i + self.processing_batch_size, self.predictions.shape[-2]))
            )
            temp_predictions = self.predictions[:, range_idx, :]
            if not self.use_logits:
                temp_predictions = torch.nn.Softmax(dim=-1)(temp_predictions)
            temp_predictions[torch.isnan(temp_predictions)] = 0


            isposinf = torch.isposinf(temp_predictions)
            isneginf = torch.isneginf(temp_predictions)

            imputation_value_posinf = temp_predictions[~isposinf].max()
            imputation_value_neginf = temp_predictions[~isneginf].min()

            #TODO: User torch.nan_to_num
            temp_predictions[isposinf] = imputation_value_posinf
            temp_predictions[isneginf] = imputation_value_neginf

            if temp_predictions.shape[1] > 0:
                self.predictions[:, range_idx, :] = temp_predictions

        return self.hp_candidates, self.time, self.predictions, self.targets

    def set_state(self, dataset_name: str, 
                  split: str = "valid"):

        self.split = split
        self.dataset_name = dataset_name
        self.get_dataset_info(dataset_name)
        self.get_statistics_dataset()

        super().set_state(dataset_name=dataset_name,
                            split=split)


    def get_dataset_names(self) -> list[str]:
        dataset_names = os.listdir(os.path.join(self.data_dir, "per_dataset"))
        dataset_names = [
            dataset for dataset in dataset_names if self.data_version in dataset
        ]
        return dataset_names
    
    def get_targets(self):
        return self.targets

    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hp_candidates, self.hp_candidates_ids

    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:
        return torch.nn.Softmax(dim=-1)(self.predictions[torch.LongTensor(ensembles)])

    def get_num_samples(self):
        return self.predictions.shape[1]

    def get_num_classes(self) -> int:
        return self.predictions.shape[-1]
    
    def get_num_pipelines(self) -> int:
        return len(self.hp_candidates_ids)

    def get_features(self, ensembles: list[list[int]]) -> torch.Tensor:
        return self.hp_candidates[torch.LongTensor(ensembles)]
    
    def get_time(self, ensembles: list[list[int]]) -> torch.Tensor:
        time_per_pipeline = self.time[torch.LongTensor(ensembles)]
        return time_per_pipeline
    
    # def evaluate_ensembles(
    #     self, ensembles: list[list[int]]
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     return self._evaluate_ensembles(ensembles=ensembles, weights=None)

    # def evaluate_ensembles_with_weights(
    #     self, ensembles: list[list[int]], weights: list[float]
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     return self._evaluate_ensembles(ensembles=ensembles, weights=weights)

    # def _evaluate_ensembles(self, ensembles: list[list[int]], weights: torch.Tensor):
    #     batch_size = len(ensembles)
    #     ensembles = torch.LongTensor(ensembles).to(self.device)
    #     # hp_candidates, time, predictions, targets = self.get_dataset_info(self.dataset_name)

    #     time_per_pipeline = self.time[ensembles]

    #     # Predictions shape: [Num. ensembles X Num. pipelines X Num Samples X Num. Classes]
    #     # predictions = self.predictions[ensembles].to(self.device)

    #     predictions = self.get_predictions(ensembles).to(self.device)

    #     targets = torch.tile(self.targets.unsqueeze(0), (batch_size, 1)).to(self.device)
    #     hp_candidates = self.hp_candidates[ensembles]
    #     metric = []
    #     metric_per_pipeline = []
    #     total_num_samples = predictions.shape[-2]
    #     for i in range(0, total_num_samples, self.processing_batch_size):
    #         range_idx = list(
    #             range(i, min(i + self.processing_batch_size, total_num_samples))
    #         )
    #         current_samples_ratio = len(range_idx) / total_num_samples
    #         temp_predictions = predictions[..., range_idx, :]
    #         temp_targets = targets[:, range_idx]
    #         if weights is not None:
    #             temp_weights = weights[..., range_idx, :]
    #         else:
    #             temp_weights = None
    #         temp_metric, temp_metric_per_pipeline = self._compute_metrics(
    #             temp_predictions, temp_targets, temp_weights, batch_size
    #         )
    #         metric.append(temp_metric.unsqueeze(-1) * current_samples_ratio)
    #         metric_per_pipeline.append(
    #             temp_metric_per_pipeline.unsqueeze(-1) * current_samples_ratio
    #         )

    #     metric = torch.cat(metric, axis=-1).sum(-1)
    #     metric_per_pipeline = torch.cat(metric_per_pipeline, axis=-1).sum(-1)

    #     return hp_candidates, metric, metric_per_pipeline, time_per_pipeline


    # def _compute_metrics(
    #     self,
    #     predictions: torch.Tensor,
    #     targets: torch.Tensor,
    #     weights: torch.Tensor,
    #     batch_size: int,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     n_classes = predictions.shape[-1]
    #     ensemble_size = predictions.shape[1]
    #     temp_targets = torch.tile(targets.unsqueeze(1), (1, ensemble_size, 1))

    #     cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    #     if weights is not None:
    #         assert weights.shape == predictions.shape
    #         weights = weights.to(self.device)
    #         weighted_predictions = torch.multiply(predictions, weights)
    #     else:
    #         weighted_predictions = predictions

    #     if self.metric_name == "error":
    #         metric_per_sample = torch.ne(
    #             predictions.reshape(-1, n_classes).argmax(-1), temp_targets.reshape(-1)
    #         ).float()
    #         metric_per_sample = metric_per_sample.reshape(batch_size, ensemble_size, -1)
    #         metric_per_pipeline = metric_per_sample.mean(axis=2)
    #         metric_ensemble_per_sample = torch.ne(
    #             weighted_predictions.sum(1).reshape(-1, n_classes).argmax(-1),
    #             targets.reshape(-1),
    #         ).float()
    #         metric = metric_ensemble_per_sample.reshape(batch_size, -1).mean(axis=-1)

    #     elif self.metric_name == "nll":
    #         logits = self.get_logits_from_probabilities(predictions)
    #         metric_per_sample = cross_entropy(
    #             logits.reshape(-1, n_classes), temp_targets.reshape(-1)
    #         )
    #         metric_per_sample = metric_per_sample.reshape(batch_size, ensemble_size, -1)
    #         metric_per_pipeline = metric_per_sample.mean(axis=-1)

    #         # logits
    #         weighted_logits = self.get_logits_from_probabilities(
    #             weighted_predictions.sum(1)
    #         )
    #         metric_ensemble_per_sample = cross_entropy(
    #             weighted_logits.reshape(-1, n_classes), targets.reshape(-1)
    #         )
    #         metric = metric_ensemble_per_sample.reshape(batch_size, -1).mean(axis=-1)

    #     else:
    #         raise ValueError("metric_name must be either error or nll")

    #     return metric, metric_per_pipeline
