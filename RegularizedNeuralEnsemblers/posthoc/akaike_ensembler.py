from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..samplers.random_sampler import RandomSampler
from .base_ensembler import BaseEnsembler


class AkaikeEnsembler(BaseEnsembler):
    """Ensembler using akaike-weighting or pseudo-model averaging."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)

    def get_akaike_weights(self, metrics_per_pipeline):
        # implementing pseudo bayesian model averaging: https://www.pymc.io/projects/examples/en/stable/diagnostics_and_criticism/model_averaging.html
        # Wagenmakers, E. J., & Farrell, S. (2004). AIC model selection using Akaike weights. Psychonomic bulletin & review, 11, 192-196.

        lowest_metric = metrics_per_pipeline.min()
        d_metric = metrics_per_pipeline - lowest_metric
        weights = torch.nn.Softmax()(-(1 / 2) * d_metric)
        return weights

    def sample(
        self,
        X_obs,
        **kwargs,  # pylint: disable=unused-argument
    ) -> tuple[list, float]:
        """Sample from the ensembler."""
        max_num_pipelines = kwargs.get("max_num_pipelines", -1)

        self.X_obs = X_obs
        self.best_ensemble = X_obs

        metrics_per_pipeline = self.metadataset.evaluate_ensembles([X_obs])[2][0]

        if max_num_pipelines != -1:
            cutoff = metrics_per_pipeline.sort()[0][max_num_pipelines]
            metrics_per_pipeline[metrics_per_pipeline > cutoff] = torch.inf

        self.weights = self.get_akaike_weights(metrics_per_pipeline)
        y_pred = self.compute_predictions()
        y_true = self.metadataset.get_targets()
        metric = self.metadataset.score_y_pred(y_pred, y_true)

        if self.normalize_performance:
            metric = self.metadataset.normalize_performance(metric)

        return self.best_ensemble, metric

    def compute_predictions(self):
        base_functions = (
            self.metadataset.get_predictions([self.X_obs])[0]
            .transpose(0, 1)
            .transpose(2, 1)
        )

        weights = torch.repeat_interleave(
            self.weights.unsqueeze(0), base_functions.shape[0], dim=0
        )
        weights = torch.repeat_interleave(
            weights.unsqueeze(1), base_functions.shape[1], dim=1
        )

        y_pred = torch.multiply(base_functions, weights).sum(axis=-1)
        return y_pred

    def evaluate_on_split(self, split: str = "test"):
        self.metadataset.set_state(
            dataset_name=self.metadataset.dataset_name, split=split
        )
        base_functions = (
            self.metadataset.get_predictions([self.best_ensemble])[0]
            .transpose(0, 1)
            .transpose(2, 1)
        ).numpy()
        y_pred = self.compute_predictions()
        y_true = self.metadataset.get_targets()
        metric = self.metadataset.score_y_pred(y_pred, y_true)

        if self.normalize_performance:
            metric = self.metadataset.normalize_performance(metric)

        return metric
