from __future__ import annotations

import numpy as np
import torch

from ..utils.common import move_to_device
from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(
        self, metadataset, device: torch.device = torch.device("cpu"), patience: int = 50
    ):
        super().__init__(metadataset=metadataset, patience=patience, device=device)

    @move_to_device
    def sample(
        self,
        acquisition_function=None,  # TODO: remove acquisition_function
        max_num_pipelines: int = 10,
        batch_size: int = 16,
        observed_pipeline_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observed_pipeline_ids is None or len(observed_pipeline_ids) == 0:
            candidates = self.metadataset.hp_candidates_ids.numpy()
        else:
            candidates = np.array(observed_pipeline_ids)

        num_pipelines = np.random.randint(1, max_num_pipelines + 1)
        ensembles = np.random.randint(0, len(candidates), (batch_size, num_pipelines))
        ensembles = candidates[ensembles].tolist()

        (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            time_per_pipeline,
        ) = self.metadataset.evaluate_ensembles(ensembles=ensembles)

        # TODO: return ensembles to get the ids
        return pipeline_hps, metric, metric_per_pipeline, time_per_pipeline
