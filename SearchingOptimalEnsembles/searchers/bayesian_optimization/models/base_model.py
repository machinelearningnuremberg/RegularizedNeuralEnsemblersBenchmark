from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb

from ....samplers.base_sampler import BaseSampler
from ....utils.logger import get_logger
from .utils import Checkpointer


class BaseModel(nn.Module):
    class ModelCheckpointer(Checkpointer):
        def __init__(self, model, checkpoint_path: Path | None = None):
            super().__init__(checkpoint_path)
            self.model = model

        def _load_checkpoint(self, checkpoint: Path = Path("surrogate.pth")):
            """Loads the checkpoint from the checkpoint path."""
            self.model.load_checkpoint(checkpoint)

        def _save_checkpoint(self, checkpoint: Path = Path("surrogate.pth")):
            """Saves the checkpoint to the checkpoint path."""
            self.model.save_checkpoint(checkpoint)

    def __init__(
        self,
        sampler: BaseSampler,
        add_y: bool = True,
        checkpoint_path: Path | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.sampler = sampler
        self.add_y = add_y
        self.dim_in = self.sampler.metadataset.feature_dim
        #assert self.dim_in is not None, "Feature dimension is None"
        if self.add_y:
            self.dim_in += 2  # counting y and the mask

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.logger = get_logger(name="SEO-MODEL", logging_level="debug")

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.default_config: dict[str, Any]
        self.checkpointer = self.ModelCheckpointer(self, checkpoint_path)

        self.observed_pipeline_ids: list[int] | None = None

    def fit(
        self,
        num_epochs: int = 100,
        observed_pipeline_ids: list[int] | None = None,
        max_num_pipelines: int = 10,
        batch_size: int = 16,
    ) -> float | None:
        """
        Trains or evaluates the model based on a dataset.
        This function computes the loss for either the training or evaluation mode.

        Args:
            num_epochs (int, optional): Number of training epochs for each dataset. Defaults to 100.
            observed_pipeline_ids (list[int], optional): List of pipeline ids that have been observed. Defaults to None.
            max_num_pipelines (int, optional): Maximum number of pipelines to sample. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 16.
        Returns:
            float: The average loss across all the batches and datasets.

        Raises:
            Exception: Catches any exception that occurs during batch training and logs it.
        """

        self.observed_pipeline_ids = observed_pipeline_ids

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
            ) = self.sampler.sample(
                observed_pipeline_ids=observed_pipeline_ids,
                max_num_pipelines=max_num_pipelines,
                batch_size=batch_size,
            )

            try:
                loss = self._fit_batch(
                    pipeline_hps=pipeline_hps,
                    metric_per_pipeline=metric_per_pipeline,
                    metric=metric,
                )

                # Logging the loss for the current iteration
                if num_epochs > 1:
                    self.logger.debug(
                        f"Inner loop step {epoch+1}/{num_epochs} - Loss: {float(loss):.5f}"
                    )

            except Exception as e:  # Handle exceptions
                self.logger.debug(
                    f"Epoch {epoch+1}/{num_epochs} - Exception during training: {e}"
                )
                continue

            if wandb.run is not None:
                wandb.log({"meta_meta_train_loss": loss})

        return loss.item() if loss is not None else None

    def _mask_y(self, y, shape):
        if y is None:
            y = torch.zeros(shape[0], shape[1], 1).to(self.device)
            mask = y
        else:
            ones_pct = 1 - 1 / shape[1]
            y_temp = y.unsqueeze(-1)
            mask = torch.bernoulli(torch.full(y_temp.shape, ones_pct)).to(self.device)

            if torch.sum(mask) == 0:
                mask = torch.ones(mask.shape).to(self.device)

            y_temp = y_temp.to(self.device)
            y = y_temp * mask
        return y, mask.bool()

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
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and standard deviation of the predictive distribution

        Args:
            x (torch.Tensor):
                The input tensor to evaluate the model.
                    Shape should be (B, N, F) where:
                    - B is the batch size
                    - N is the number of pipelines
                    - F is the number of features

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the
                predictive distribution
        """

        raise NotImplementedError
