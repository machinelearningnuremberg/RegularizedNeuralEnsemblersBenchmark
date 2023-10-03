from __future__ import annotations

import itertools

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_sampler import BaseSampler
from .random_sampler import RandomSampler


class DiversitySampler(RandomSampler):
    def __init__(
        self,
        metadataset: BaseMetaDataset,
        patience: int = 50,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(metadataset=metadataset, patience=patience, device=device)
        self.metadataset = metadataset

    def diversity_score(self, predictions1, predictions2):
        return torch.sqrt(((predictions1 - predictions2) ** 2).sum(-1)).mean(-1) * (
            1 / np.sqrt(2)
        )

    def sample_with_diversity_score(
        self,
        batch_size: int = 16,
        observed_pipeline_ids: list[int] | None = None,
    ) -> tuple[np.array, np.array, np.array]:
        if observed_pipeline_ids is None:
            pipeline_hps, _, _, _, ensembles = super().sample(
                max_num_pipelines=1, batch_size=batch_size
            )

        else:
            ensembles = [[x] for x in observed_pipeline_ids]
            pipeline_hps, _, _, _ = self.metadataset.evaluate_ensembles(
                ensembles=ensembles
            )

        num_pipelines = len(ensembles)
        pairs_list = list(itertools.product(range(num_pipelines), range(num_pipelines)))
        np.random.shuffle(pairs_list)
        pairs = torch.LongTensor(pairs_list)[:batch_size]

        predictions = self.metadataset.get_predictions(ensembles=ensembles).squeeze(1)
        predictions1 = predictions[pairs[:, 0]]
        predictions2 = predictions[pairs[:, 1]]
        pipeline_hps1 = pipeline_hps[pairs[:, 0]].squeeze(1)
        pipeline_hps2 = pipeline_hps[pairs[:, 1]].squeeze(1)

        scores = self.diversity_score(predictions1, predictions2)

        return pipeline_hps1, pipeline_hps2, scores
