from __future__ import annotations

import logging
from abc import abstractmethod

import torch


class BaseOptimizer:
    """Base optimizer class."""

    def __init__(
        self,
        metadataset,
        patience: int = 50,
        logger=None,
        budget: None | int | float = None,
    ):
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.metadataset = metadataset
        self.patience = patience
        self.logger = logger or logging.getLogger("seo")
        self.budget = budget
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.logger.info(f"Using device: {self.device}")

    @abstractmethod
    def run(self):
        raise NotImplementedError
