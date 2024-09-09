from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_ensembler import BaseEnsembler


class SingleBest(BaseEnsembler):
    """Single Best (no ensemble) class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)

    def sample(self,
               X_obs,
               **kwargs) -> tuple[list, float]:

        """Return single best on the validation performance"""

        self.X_obs = X_obs
        ensembles = [[x] for x in X_obs]
        _, metric, _, _ = self.metadataset.evaluate_ensembles(ensembles)
        metric = metric.numpy()
        best_score = np.min(metric)
        best_ones = np.array(ensembles)[np.where(metric==best_score)]
        best_ensemble = [np.random.choice(best_ones.flatten())]

        self.best_ensemble = best_ensemble
        return best_ensemble, best_score
