from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..utils.common import move_to_device
from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(
        self,
        metadataset: BaseMetaDataset,
        patience: int = 50,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(metadataset=metadataset, patience=patience, device=device)

    def generate_ensembles(
        self,
        candidates: np.ndarray,
        num_pipelines: int = 10,
        batch_size: int = 16,
    ) -> list[list[int]]:
        ensembles = np.random.randint(0, len(candidates), (batch_size, num_pipelines))
        return candidates[ensembles].tolist()

    @move_to_device
    def sample(
        self,
        max_num_pipelines: int = 10,
        fixed_num_pipelines: int | None = None,
        batch_size: int = 16,
        observed_pipeline_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
        if observed_pipeline_ids is None or len(observed_pipeline_ids) == 0:
            candidates = self.metadataset.hp_candidates_ids.numpy().astype(int)
        else:
            candidates = np.array(observed_pipeline_ids)

        if fixed_num_pipelines is not None:
            num_pipelines = fixed_num_pipelines
        else:
            num_pipelines = np.random.randint(1, max_num_pipelines + 1)
        ensembles = np.random.randint(0, len(candidates), (batch_size, num_pipelines))
        ensembles = candidates[ensembles].tolist()

        (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            time_per_pipeline,
        ) = self.metadataset.evaluate_ensembles(ensembles=ensembles)

        return pipeline_hps, metric, metric_per_pipeline, time_per_pipeline, ensembles
