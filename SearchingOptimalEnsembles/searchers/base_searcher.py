from __future__ import annotations

from abc import abstractmethod

import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..utils.logger import get_logger


class BaseSearcher:
    """Base searcher class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
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

        self.X_obs = None
        self.X_pending = None
        self.incumbent = None

        self.metadataset = metadataset
        self.patience = patience
        self.logger = get_logger(name="SEO-SEARCHER", logging_level="debug")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.debug(f"Using device: {self.device}")

    def set_state(
        self,
        X_obs: torch.Tensor,
        X_pending: torch.Tensor,
        incumbent: torch.Tensor | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """Set the state of the searcher.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self.X_obs = X_obs
        self.X_pending = X_pending
        self.incumbent = incumbent

    @abstractmethod
    def suggest(self, **kwargs):
        raise NotImplementedError
