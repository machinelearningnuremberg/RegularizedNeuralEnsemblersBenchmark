from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch


class BaseSampler:
    def __init__(
        self, metadataset, patience: int = 50, device: torch.device = torch.device("cpu")
    ):
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.metadataset = metadataset
        self.acquisition_function = None
        self.patience = patience
        self.device = device

        # To initialize call set_state() in the child class
        self.dataset_name: str

    @abstractmethod
    def sample(
        self,
        acquisition_function=None,
        max_num_pipelines: int = 10,
        batch_size: int = 16,
        observed_pipeline_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def set_state(
        self, dataset_name: str | None = None, meta_split: str = "meta-train"
    ) -> None:
        """Set the state of the sampler. This includes populating the dataset name.
        If dataset_name is None, a random dataset is chosen from the specified
        meta-split.

        Args:
            dataset_name (str, optional): Name of the dataset. Defaults to None.
            meta_split (str, optional): Meta-split name. Defaults to "meta-train".
        """

        if dataset_name is None:
            dataset_name = np.random.choice(self.metadataset.meta_splits[meta_split])

        self.dataset_name = dataset_name
        if hasattr(self.metadataset, "set_dataset"):
            self.metadataset.set_dataset(dataset_name=dataset_name)
