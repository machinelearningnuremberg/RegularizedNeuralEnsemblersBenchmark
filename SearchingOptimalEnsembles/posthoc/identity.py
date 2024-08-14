from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..samplers.random_sampler import RandomSampler
from .base_ensembler import BaseEnsembler


class IdentityEnsembler(BaseEnsembler):
    """Random ensembler class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)

        

    def sample(
        self,
        X_obs,
        **kwargs,  # pylint: disable=unused-argument
    ) -> tuple[list, float]:
        """Sample from the ensembler."""

        self.X_obs = X_obs
        self.best_ensemble = X_obs

        _, best_score, _, _ = self.metadataset.evaluate_ensembles([X_obs])

        return self.best_ensemble, best_score
