# based on NES: https://github.com/arberzela/nes-bo/blob/main/nes/nasbench201/scripts/run_nes_bo_sls_evo.py

from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_sampler import BaseSampler


class LocalSearchSampler(BaseSampler):
    def __init__(
        self,
        metadataset: BaseMetaDataset,
        patience: int = 50,
        device: torch.device = torch.device("cpu"),
        #############################
        ls_iter: int = 10,
    ):
        super().__init__(metadataset=metadataset, patience=patience, device=device)
        self.ls_iter = ls_iter

    def search_ensemble(
        self,
        candidates: np.ndarray,
        num_pipelines: int = 10,
    ) -> list[int]:
        choices_idx = np.random.choice(len(candidates), num_pipelines, replace=False)
        best_X = choices_idx.tolist()
        best_y = self.metadataset.evaluate_ensembles(ensembles=[best_X])[1]

        for _ in range(self.ls_iter):
            ids = list(set(np.arange(len(candidates))) - set(choices_idx))
            new_sampled_pipeline = np.random.choice(ids)
            random_id = np.random.choice(len(choices_idx))
            new_choices_idx = np.copy(choices_idx)
            new_choices_idx[random_id] = new_sampled_pipeline
            ens_to_evaluate = [candidates[i] for i in new_choices_idx]

            # pylint: disable=unused-variable
            (
                pipeline_hps,
                metric,
                metric_per_pipeline,
                time_per_pipeline,
            ) = self.metadataset.evaluate_ensembles(ensembles=[ens_to_evaluate])

            if metric < best_y:
                best_y = metric
                best_X = ens_to_evaluate
                choices_idx = new_choices_idx

        return best_X

    def generate_ensembles(
        self, candidates: np.ndarray, num_pipelines: int = 10, batch_size: int = 16
    ) -> list[list[int]]:
        ensembles = []
        for _ in range(batch_size):
            new_ensemble = self.search_ensemble(candidates, num_pipelines)
            ensembles.append(new_ensemble)

        return ensembles
