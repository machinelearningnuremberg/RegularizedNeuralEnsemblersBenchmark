from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_ensembler import BaseEnsembler


class QuickGreedyEnsembler(BaseEnsembler):
    """Quick and Greedy ensembler class. Starting with the best
    pipeline by validation performance, add the next best pipeline
    to the ensemble only if it improves validation performance,
    iterating until the ensemble size is M or all pipelines have been
    considered (returning an ensemble of size at most M."""

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
        ensemble: list[int] = []
        best_metric = torch.inf

        # TODO: Fix the type of X_obs
        X_obs = np.array(X_obs).reshape(-1, 1)

        # evaluate ensembles with only one pipeline to get single pipeline
        # metrics
        temp_metric = self.metadataset.evaluate_ensembles(ensembles=X_obs.tolist())[1]

        ranked_ids = temp_metric.argsort().numpy()
        ranked_X_obs = X_obs[ranked_ids].tolist()
        ensemble.append(ranked_X_obs[0][0])

        num_skipped = 0
        for i in range(1, len(X_obs)):
            print(f"QuickGreedyEnsembler: {i}")
            temp_ensemble = ensemble + ranked_X_obs[i]
            (
                _,
                metric,
                _,
                _,
            ) = self.metadataset.evaluate_ensembles(ensembles=[temp_ensemble])

            if metric < best_metric:
                ensemble = temp_ensemble
                best_metric = metric
            else:
                num_skipped += 1

            if len(ensemble) == max_num_pipelines:
                break

        print(f"Number of skipped pipelines: {num_skipped}")
        if len(ensemble) < max_num_pipelines:
            print(f"Ensemble returned has size {len(ensemble)} < max_num_pipelines.")
        self.best_ensemble = ensemble
        return ensemble, best_metric
