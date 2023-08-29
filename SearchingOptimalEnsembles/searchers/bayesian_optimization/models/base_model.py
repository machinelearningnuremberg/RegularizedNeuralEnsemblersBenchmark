from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

from ....samplers.base_sampler import BaseSampler
from ....utils.logger import get_logger


class BaseModel(nn.Module):
    def __init__(
        self,
        sampler: BaseSampler,
        checkpoint_path: Path | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.sampler = sampler
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.logger = get_logger(name="SEO-MODEL", logging_level="debug")

        self.observed_ids = None
        self.x_obs: list[torch.Tensor] = []
        self.y_obs: list[torch.Tensor] = []

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer

    @abstractmethod
    def load_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """Loads the checkpoint from the checkpoint path."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """Saves the checkpoint to the checkpoint path."""
        raise NotImplementedError

    @abstractmethod
    def checkpoint_exists(self, checkpoint_name="checkpoint.pth") -> bool:
        """Checks if the checkpoint exists.

        Args:
            checkpoint_name (str, optional): Name of the checkpoint. Defaults to "checkpoint.pth".

        Returns:
            bool: True if the checkpoint exists.
        """
        raise NotImplementedError

    def _observe(self, x: torch.Tensor, y: torch.Tensor):
        self.x_obs.append(x)
        self.y_obs.append(y)

    def fit(
        self,
        num_epochs: int = 100,
    ) -> float | None:
        """
        Trains or evaluates the model based on a dataset.
        This function computes the loss for either the training or evaluation mode.

        Args:
            num_epochs (int, optional): Number of training epochs for each dataset. Defaults to 100.
        Returns:
            float: The average loss across all the batches and datasets.

        Raises:
            Exception: Catches any exception that occurs during batch training and logs it.
        """
        loss = None

        # Loop for each epoch
        for epoch in range(num_epochs):
            # pylint: disable=unused-variable
            (
                pipeline_hps,
                metric,
                metric_per_pipeline,
                time_per_pipeline,
                ensembles,
            ) = self.sampler.sample(observed_pipeline_ids=self.observed_ids)

            try:
                loss = self._fit_batch(
                    pipeline_hps=pipeline_hps,
                    metric_per_pipeline=metric_per_pipeline,
                    metric=metric,
                )

                # Logging the loss for the current iteration
                if num_epochs > 1:
                    self.logger.debug(
                        f"Inner loop step {epoch+1}/{num_epochs} - Loss: {loss.item():.5f}"
                    )

            except Exception as e:  # Handle exceptions
                self.logger.debug(
                    f"Epoch {epoch+1}/{num_epochs} - Exception during training: {e}"
                )
                continue

        return loss

    @abstractmethod
    def _fit_batch(
        self,
        pipeline_hps: torch.Tensor,
        metric_per_pipeline: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        """Fits the model to the observed data. Returns the loss and the noise
        of the likelihood.

        Args:
            pipeline_hps (torch.Tensor):
                 Hyperparameters of the pipelines as tensor.
                    Shape should be (B, N, F) where:
                    - B is the batch size
                    - N is the number of pipelines
                    - F is the number of features
            metric (torch.Tensor): Metric of the pipelines, shape (B, N) where B is the
            batch size and N is the number of pipelines.
            optimizer (torch.optim.Optimizer): Optimizer to use for training.
            mode (str, optional): Mode of the model. Defaults to "train". Can be "train"
                or "eval".  If "eval", the loss is calculated as MSE.

        Returns:
            torch.Tensor: Loss.
        """

        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        pipeline_hps: torch.Tensor,
        metric_per_pipeline: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        """Validates the model to the observed data. Returns the loss and the noise
        of the likelihood.

        Args:
            pipeline_hps (torch.Tensor):
                 Hyperparameters of the pipelines as tensor to validate the model.
                    Shape should be (B, N, F) where:
                    - B is the batch size
                    - N is the number of pipelines
                    - F is the number of features
            metric (torch.Tensor): Metric of the pipelines, shape (B, N) where B is the
            batch size and N is the number of pipelines.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, x: torch.Tensor, sampler: BaseSampler, max_num_pipelines: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and standard deviation of the predictive distribution

        Args:
            x (torch.Tensor):
                The input tensor to evaluate the model.
                    Shape should be (B, N, F) where:
                    - B is the batch size
                    - N is the number of pipelines
                    - F is the number of features
            sampler (BaseSampler): Sampler to generate candidates.
            max_num_pipelines (int, optional): Maximum number of pipelines in the
                ensemble. Defaults to 10.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the
                predictive distribution
        """

        raise NotImplementedError
