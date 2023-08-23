from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ....samplers.base_sampler import BaseSampler


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

        self.observed_ids = None
        self.x_obs: list[torch.Tensor] = []
        self.y_obs: list[torch.Tensor] = []

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer

    @abstractmethod
    def load_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        raise NotImplementedError

    def _observe(self, x: torch.Tensor, y: torch.Tensor):
        self.x_obs.append(x)
        self.y_obs.append(y)

    def train(
        self,
        training_epochs: int = 100,
        max_patience: int = 16,
        loss_tol: float = 0.0001,
    ):
        def print_progress(iteration: int, loss: float, noise: float):
            logging.debug(
                f"Iter {iteration}/{training_epochs} - Loss: {loss:.5f} noise: {noise:.5f}"
            )

        self.load_checkpoint()

        losses = [np.inf]
        best_loss = np.inf
        weights = copy.deepcopy(self.state_dict())

        self.model.train()

        for epoch in range(training_epochs):
            (
                pipeline_hps,
                metric,
                metric_per_pipeline,  # pylint: disable=unused-variable
                time_per_pipeline,  # pylint: disable=unused-variable
            ) = self.sampler.sample(observed_pipeline_ids=self.observed_ids)
            # TODO: get the ensembles

            try:
                loss, noise = self._fit_batch(
                    pipeline_hps=pipeline_hps,
                    metric=metric,
                    optimizer=self.optimizer,
                )
            except Exception as e:
                logging.info(f"Exception {e}")
                continue  # TODO: handle this better

            # self._observe(x=pipeline_hps, y=metric)

            print_progress(epoch + 1, loss.item(), noise.item())

            losses.append(loss.detach().to("cpu").item())
            if best_loss > losses[-1]:
                best_loss = losses[-1]
                weights = copy.deepcopy(self.state_dict())

            epochs = training_epochs
            # TODO: make conditional (this is finetune), do not stop for meta training, based on MSE of prediction
            if np.allclose(losses[-1], losses[-2], atol=loss_tol):
                if epoch - (epochs - max_patience) >= 0:
                    break
            else:
                epochs = min(epochs, epoch + max_patience + 1)

            # TODO: add meta-validate + early stopping here (from above)

        self.load_state_dict(weights)
        self.save_checkpoint()

    @abstractmethod
    def _fit_batch(
        self,
        pipeline_hps: torch.Tensor,
        metric: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        "Fits the model to the observed data. Returns the loss and the noise"

        raise NotImplementedError

    @abstractmethod
    def predict(
        self, x: torch.Tensor, sampler: BaseSampler
    ) -> tuple[torch.Tensor, torch.Tensor]:
        "Returns the mean and standard deviation of the predictive distribution"

        raise NotImplementedError
