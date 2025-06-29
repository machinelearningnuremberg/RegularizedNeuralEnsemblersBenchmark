from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from .base_metadataset import BaseMetaDataset

class Evaluator(BaseMetaDataset):

    def __init__(
        self,
        data_dir: str,
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "nll",
        device: torch.device = torch.device("cpu"),
        processing_batch_size: int = 1000,
        pct_valid_data: float = 1.,
        data_version: str = None
    ):

        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
            data_version=data_version
        )
        self.device = device
        self.processing_batch_size = processing_batch_size
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
        self.pct_valid_data = pct_valid_data

        #following: https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_percentage_error.html
        self.absolute_relative_error = lambda y_true,y_pred: torch.abs(y_true-y_pred)/torch.max(torch.ones(1),torch.abs(y_true))
        self.mse = torch.nn.MSELoss(reduction="none")

        #return torch.zeros(len(ensembles),
        #                    len(ensembles[0]))

    def evaluate_ensembles(
        self, ensembles: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_ensembles(ensembles=ensembles, weights=None)

    def evaluate_ensembles_with_weights(
        self, ensembles: list[list[int]], weights: list[float]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_ensembles(ensembles=ensembles, weights=weights)

    def subsample(self, predictions, targets):
        if self.split in ["valid", "val"] and self.pct_valid_data<1.:
            valid_data_size = max(int(self.pct_valid_data*predictions.shape[2]),predictions.shape[3] )
            idx = np.random.default_rng(seed=self.seed).integers(0, predictions.shape[2], valid_data_size)
            return predictions[..., idx,:], targets[idx]
        else:
            return predictions, targets

    def _evaluate_ensembles(self, ensembles: list[list[int]], weights: torch.Tensor):
        batch_size = len(ensembles)

        # Predictions shape: [Num. ensembles X Num. pipelines X Num Samples X Num. Classes]
        time_per_pipeline = self.get_time(ensembles).to(self.device)
        predictions = self.get_predictions(ensembles).to(self.device)
        targets = self.get_targets().to(self.device)
        predictions, targets = self.subsample(predictions, targets)
        targets = torch.tile(targets, (batch_size, 1)).to(self.device)
        hp_candidates = self.get_features(ensembles)

        ensembles = torch.LongTensor(ensembles).to(self.device)

        metric = []
        metric_per_pipeline = []
        total_num_samples = predictions.shape[-2]
        for i in range(0, total_num_samples, self.processing_batch_size):
            range_idx = list(
                range(i, min(i + self.processing_batch_size, total_num_samples))
            )
            current_samples_ratio = len(range_idx) / total_num_samples
            temp_predictions = predictions[..., range_idx, :]
            temp_targets = targets[:, range_idx]
            if weights is not None:
                temp_weights = weights[..., range_idx, :]
            else:
                temp_weights = None
            temp_metric, temp_metric_per_pipeline = self._compute_metrics(
                temp_predictions, temp_targets, temp_weights, batch_size
            )
            metric.append(temp_metric.unsqueeze(-1) * current_samples_ratio)
            metric_per_pipeline.append(
                temp_metric_per_pipeline.unsqueeze(-1) * current_samples_ratio
            )

        metric = torch.cat(metric, axis=-1).sum(-1)
        metric_per_pipeline = torch.cat(metric_per_pipeline, axis=-1).sum(-1)

        return hp_candidates, metric, metric_per_pipeline, time_per_pipeline


    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_classes = predictions.shape[-1]
        ensemble_size = predictions.shape[1]
        temp_targets = torch.tile(targets.unsqueeze(1), (1, ensemble_size, 1))

        if weights is not None:
            assert weights.shape == predictions.shape
        else:
            weights = torch.ones_like(predictions).div(ensemble_size)
        weights = weights.to(self.device)
        weighted_predictions = torch.multiply(predictions, weights)

        if self.metric_name == "error":
            metric_per_sample = torch.ne(
                predictions.reshape(-1, n_classes).argmax(-1), temp_targets.reshape(-1)
            ).float()
            metric_per_sample = metric_per_sample.reshape(batch_size, ensemble_size, -1)
            metric_per_pipeline = metric_per_sample.mean(axis=2)
            metric_ensemble_per_sample = torch.ne(
                weighted_predictions.sum(1).reshape(-1, n_classes).argmax(-1),
                targets.reshape(-1),
            ).float()
            metric = metric_ensemble_per_sample.reshape(batch_size, -1).mean(axis=-1)

        elif self.metric_name == "absolute_relative_error":
            metric_per_sample = self.absolute_relative_error(temp_targets,predictions.squeeze(-1))
            metric_per_pipeline = metric_per_sample.mean(-1)
            metric_ensemble_per_sample = self.absolute_relative_error(temp_targets, weighted_predictions.sum(axis=1, keepdim=True).squeeze(-1))
            metric = metric_ensemble_per_sample.reshape(batch_size, -1).mean(-1)

        elif self.metric_name == "mse":
            metric_per_sample = self.mse(temp_targets,predictions.squeeze(-1))
            metric_per_pipeline = metric_per_sample.mean(-1)
            metric_ensemble_per_sample = self.mse(temp_targets, weighted_predictions.sum(axis=1, keepdim=True).squeeze(-1))
            metric = metric_ensemble_per_sample.reshape(batch_size, -1).mean(-1)

        elif self.metric_name == "neg_roc_auc":
            y_pred = predictions.mean(1)
            metric = []
            metric_per_pipeline = torch.zeros(predictions.shape[:2])
            for i in range(y_pred.shape[0]):
                if y_pred[i].shape[1] == 2:
                    y_pred_ = y_pred[i][:,1]
                else:
                    y_pred_ = y_pred[i]

                metric.append(
                    1-roc_auc_score(targets[i].numpy(), y_pred_.numpy(), multi_class="ovo")
                )
            metric = torch.FloatTensor(metric).to(predictions.device)

        elif self.metric_name == "nll":
            logits = self.get_logits_from_probabilities(predictions)
            metric_per_sample = self.cross_entropy(
                logits.reshape(-1, n_classes), temp_targets.reshape(-1)
            )
            metric_per_sample = metric_per_sample.reshape(batch_size, ensemble_size, -1)
            metric_per_pipeline = metric_per_sample.mean(axis=-1)

            # logits
            weighted_logits = self.get_logits_from_probabilities(
                weighted_predictions.sum(1)
            )
            metric_ensemble_per_sample = self.cross_entropy(
                weighted_logits.reshape(-1, n_classes), targets.reshape(-1)
            )
            metric = metric_ensemble_per_sample.reshape(batch_size, -1).mean(axis=-1)

        else:
            raise ValueError("metric_name must be either error or nll")

        return metric, metric_per_pipeline

    def score_y_pred(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: predictions (outpur porbabilities for classification).
        """
        y_true = y_true.to(y_pred.device)

        if self.metric_name == "absolute_relative_error":
            metric = self.absolute_relative_error(y_true, y_pred.reshape(-1)).mean()
        elif self.metric_name == "mse":
            metric = self.mse(y_true, y_pred.reshape(-1)).mean()
        else:
            y_true = y_true.long()
            if self.metric_name == "nll":
                y_pred = self.get_logits_from_probabilities(y_pred)
                metric = torch.nn.CrossEntropyLoss()(y_pred, y_true)
            elif self.metric_name == "error":
                metric = (y_pred.argmax(-1) != y_true).float().mean()
            elif self.metric_name == "neg_roc_auc":
                if y_pred.shape[1] == 2:
                    y_pred = y_pred[:,1]
                metric = 1-roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), multi_class="ovo")
            else:
                raise ValueError("Metric name is not known.")

        return metric
