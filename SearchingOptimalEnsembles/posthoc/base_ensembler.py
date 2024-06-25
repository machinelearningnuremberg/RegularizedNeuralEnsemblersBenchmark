from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
import numpy as np
import copy

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..metadatasets import MetaDatasetMapping
from ..metadatasets.base_metadataset import META_SPLITS, BaseMetaDataset
from ..utils.common import instance_from_map


class BaseEnsembler:
    """Base ensembler class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initializes the base ensembler class."""
        self.best_ensemble = []
        self.set_state(metadataset=metadataset,
                       device=device)

    def set_state(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        self.metadataset = metadataset
        self.device = device


    @abstractmethod
    def sample(self, X_obs: np.array, **kwargs) -> tuple[list, float]:
        """Sample from the ensembler."""
        raise NotImplementedError

    def evaluate_on_split(
        self,
        split: str = "test",
    ):
        
        #self.metadataset.set_state(dataset_name=self.metadataset.dataset_name)
        #self.set_state(metadataset=self.metadataset, device=self.device)

        weights = None
        if hasattr(self, "get_weights"):
            X_context = None
            y_context = None
            if hasattr(self, "get_context"):
                self.metadataset.set_state(dataset_name=self.metadataset.dataset_name,
                                           split = "valid")
                X_context, y_context = self.get_context(self.best_ensemble)
            weights = self.get_weights(
                self.best_ensemble, X_context=X_context, y_context=y_context
            )
        self.metadataset.set_state(dataset_name=self.metadataset.dataset_name,
                                    split = split)
        if weights is None:
            _, metric, metric_per_pipeline, _ = self.metadataset.evaluate_ensembles([self.best_ensemble])
        else:
            _, metric, metric_per_pipeline, _ = self.metadataset.evaluate_ensembles_with_weights(
                [self.best_ensemble], weights
            )
        metric = self.metadataset.normalize_performance(metric)

        return metric
