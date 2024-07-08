from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_ensembler import BaseEnsembler


class TopMEnsembler(BaseEnsembler):
    """Top M ensembler class. It just selects the top M pipelines according to
    the metric that is used to evaluate the performance and ensembels them."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
        no_resample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)
        self.no_resample = no_resample

    def sample(
        self,
        X_obs,
        # max_num_pipelines: int = 5,
        # num_batches: int = 5,
        # num_suggestions_per_batch: int = 1000,
        **kwargs,
    ) -> tuple[list, float]:
        """Sample from the ensembler."""

        self.X_obs = X_obs
        max_num_pipelines = kwargs.get("max_num_pipelines", 5)

        # TODO: Fix the type of X_obs
        X_obs = np.array(X_obs).reshape(-1, 1)

        print("Top M Ensembler")
        # evaluate ensembles with only one pipeline to get single pipeline
        # metrics
        (
            _,
            temp_metric,
            _,
            _,
        ) = self.metadataset.evaluate_ensembles(ensembles=X_obs.tolist())

        best_ids = temp_metric.argsort()[:max_num_pipelines].numpy()
        ensemble = X_obs[best_ids].flatten().tolist()

        (
            _,
            metric,
            _,
            _,
        ) = self.metadataset.evaluate_ensembles(ensembles=[ensemble])

        best_metric = metric[0]

        self.best_ensemble = ensemble
        return ensemble, best_metric
