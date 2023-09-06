from __future__ import annotations

from abc import abstractmethod

import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..utils.logger import get_logger


class BaseOptimizer:
    """Base searcher class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        worker_dir: str,
        patience: int = 50,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Initializes the base searcher class.

        Args:

            metadataset (BaseMetaDataset): The meta-dataset to be used for the optimization.
            patience (int, optional): The number of epochs to wait for the validation loss to improve.

        Raises:
            ValueError: If the patience is less than 1.
        """
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.metadataset = metadataset
        self.worker_dir = worker_dir
        self.patience = patience
        self.logger = get_logger(name="SEO-SEARCHER", logging_level="debug")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.debug(f"Using device: {self.device}")

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError
