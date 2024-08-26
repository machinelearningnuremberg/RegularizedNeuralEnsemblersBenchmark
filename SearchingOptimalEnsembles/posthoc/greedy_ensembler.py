from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_ensembler import BaseEnsembler


class GreedyEnsembler(BaseEnsembler):
    """Greedy ensembler class."""

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
        X_obs = np.array(X_obs).reshape(-1, 1).tolist()
        best_ensemble = None

        for i in range(max_num_pipelines):
            print(f"GreedyEnsembler: {i}")
            temp_ensembles = (
                np.concatenate(
                    (np.array(X_obs), np.array([ensemble] * len(X_obs))), axis=1
                )
                .astype(int)
                .tolist()
            )
            (
                _,
                metric,
                _,
                _,
            ) = self.metadataset.evaluate_ensembles(ensembles=temp_ensembles)

            temp_best_id = metric.argmin()
            temp_best_metric = metric.min()
            ensemble.append(X_obs[temp_best_id][0])

            if self.no_resample:
                X_obs.pop(temp_best_id)

            if temp_best_metric < best_metric:
                best_metric = temp_best_metric
                best_ensemble = ensemble.copy()

        self.best_ensemble = best_ensemble
        return ensemble, best_metric
