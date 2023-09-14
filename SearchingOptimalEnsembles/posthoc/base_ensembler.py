from __future__ import annotations

from abc import abstractmethod

import torch

from ..metadatasets.base_metadataset import BaseMetaDataset


class BaseEnsembler:
    """Base ensembler class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initializes the base ensembler class."""

        self.metadataset = metadataset
        self.device = device

    @abstractmethod
    def sample(self, X_obs, **kwargs) -> tuple[list, float]:
        """Sample from the ensembler."""
        raise NotImplementedError
