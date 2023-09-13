# from __future__ import annotations

# import torch

# from ..metadatasets.base_metadataset import BaseMetaDataset
# from .base_ensembler import BaseEnsembler


# class GreedyEnsembler(BaseEnsembler):
#     """Greedy ensembler class."""

#     def __init__(
#         self,
#         metadataset: BaseMetaDataset,
#         device: torch.device = torch.device("cpu"),
#     ) -> None:
#         super().__init__(metadataset=metadataset, device=device)

# def sample(
#     self,
#     X_obs,
#     # max_num_pipelines: int = 5,
#     # num_batches: int = 5,
#     # num_suggestions_per_batch: int = 1000,
#     **kwargs,
# ) -> tuple[list, float]:
#     """Sample from the ensembler."""

#     raise NotImplementedError

# best_score = np.inf
# best_ensemble = None
# for _ in range(num_batches):
#     raise NotImplementedError
# return best_ensemble, best_score
