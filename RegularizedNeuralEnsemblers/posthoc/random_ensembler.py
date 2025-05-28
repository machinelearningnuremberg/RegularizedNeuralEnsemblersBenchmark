from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..samplers.random_sampler import RandomSampler
from .base_ensembler import BaseEnsembler


class RandomEnsembler(BaseEnsembler):
    """Random ensembler class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)

        self.sampler = RandomSampler(
            metadataset=self.metadataset,
            device=self.device,
        )

    def sample(
        self,
        X_obs,
        **kwargs,  # pylint: disable=unused-argument
    ) -> tuple[list, float]:
        """Sample from the ensembler."""

        self.X_obs = X_obs

        num_suggestion_batches = kwargs.get("num_suggestion_batches", 1)
        num_suggestions_per_batch = kwargs.get("num_suggestions_per_batch", 1)
        max_num_pipelines = kwargs.get("max_num_pipelines", 5)

        best_score = np.inf
        best_ensemble = None
        for _ in range(num_suggestion_batches):
            num_pipelines = np.random.randint(1, max_num_pipelines + 1)
            ensembles = self.sampler.generate_ensembles(
                candidates=np.array(X_obs),
                num_pipelines=num_pipelines,
                batch_size=num_suggestions_per_batch,
            )

            _, metric, _, _ = self.metadataset.evaluate_ensembles(ensembles)
            temp_best_metric = metric.min()
            temp_best_id = metric.argmin()

            if temp_best_metric < best_score:
                best_score = temp_best_metric
                best_ensemble = ensembles[temp_best_id]

        assert best_ensemble is not None, "Best ensemble is None"

        self.best_ensemble = best_ensemble
        return best_ensemble, best_score
