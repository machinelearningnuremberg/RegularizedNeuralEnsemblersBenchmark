from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..utils.common import move_to_device
from .base_sampler import BaseSampler

class LocalSeaearchSampler(BaseSampler):
    def __init__(
            self,
            metadataset: BaseMetaDataset,
            patience: int = 50,
            device: torch.device = torch.device("cpu"),
        ):
            super().__init__(metadataset=metadataset, patience=patience, device=device)


    def sample(
            self,
            max_num_pipelines: int = 10,
            fixed_num_pipelines: int | None = None,
            batch_size: int = 16,
            observed_pipeline_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:

        pass