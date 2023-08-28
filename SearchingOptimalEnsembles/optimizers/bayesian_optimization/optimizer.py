from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
from typing_extensions import Literal

from ...samplers import SamplerMapping
from ...utils.common import instance_from_map
from ..base_optimizer import BaseOptimizer
from .acquisition import AcquisitionMapping
from .models import ModelMapping


class BayesianOptimization(BaseOptimizer):
    """Bayesian optimization class."""

    def __init__(
        self,
        metadataset,
        patience: int = 50,
        surrogate_name: Literal["dkl", "dre"] = "dkl",
        sampler_name: Literal["random"] = "random",
        acquisition_name: Literal["ei"] = "ei",
        initial_design_size: int = 5,
        checkpoint_path: str
        | None = "/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/checkpoints",
    ):
        """
        Initialize the Bayesian Optimization class.

        Args:
            metadataset: The meta-dataset to use for Bayesian optimization.
            patience (int, optional): The number of epochs to wait for loss improvement before early stopping. Defaults to 50.
            surrogate_name (Literal["dkl", "dre"], optional): The name of the surrogate model to use. Defaults to "dkl".
            sampler_name (Literal["random"], optional): The name of the sampler to use. Defaults to "random".
            acquisition_name (Literal["ei"], optional): The name of the acquisition function to use. Defaults to "ei".
            initial_design_size (int, optional): The number of initial design points. Defaults to 5.
            checkpoint_path (str, optional): The path to the checkpoint directory. Defaults to the specified path.

        """
        super().__init__(metadataset=metadataset, patience=patience)

        if checkpoint_path is None:
            self.checkpoint_path = Path(
                "/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/checkpoints"
            )
        else:
            self.checkpoint_path = Path(checkpoint_path)

        sampler_args = {
            "metadataset": self.metadataset,
            "device": self.device,
        }
        self.sampler = instance_from_map(
            SamplerMapping,
            sampler_name,
            name="sampler",
            kwargs=sampler_args,
        )
        surrogate_args = {
            "sampler": self.sampler,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
        }
        self.surrogate = instance_from_map(
            ModelMapping,
            surrogate_name,
            name="surrogate",
            kwargs=surrogate_args,
        )
        acquisition_args = {
            "device": self.device,
        }
        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition_name,
            name="acquisition",
            kwargs=acquisition_args,
        )

        self.initial_design_size = initial_design_size

        self.logger.debug("Initialized Bayesian optimization")
        self.logger.debug(f"Surrogate name: {surrogate_name}")
        self.logger.debug(f"Sampler name: {sampler_name}")
        self.logger.debug(f"Acquisition name: {acquisition_name}")

    # TODO: add early stopping
    def meta_train_surrogate(
        self,
        num_epochs: int = 10,
        num_inner_epochs: int = 1,
        max_patience: int = 16,
        loss_tol: float = 0.0001,
        valid_frequency: int = 100,
    ) -> None:
        """
        Perform meta-training on the surrogate model.

        This training includes both a meta-train and meta-validation phase to adjust the model's
        weights for generalized learning.

        Args:
            num_epochs (int, optional): Total number of meta-training epochs. Defaults to 100.
            num_inner_epochs (int, optional): Number of epochs for each inner optimization loop.
                These inner epochs train on individual datasets. Defaults to 100.
            max_patience (int, optional): Maximum number of epochs to continue without loss improvement.
                Useful for early stopping. Defaults to 16.
            loss_tol (float, optional): Tolerance for loss improvement. Considered as no improvement
                if less than this value. Defaults to 0.0001.
            valid_frequency (int, optional): Frequency of performing meta-validation, in terms of epochs.
                Defaults to 100000.

        Returns:
            None: Updates the model's state in-place and saves the best state to a checkpoint file.
        """

        # Initialize the surrogate model
        self.surrogate.load_checkpoint(checkpoint_name="surrogate.pth")

        # Initialize the learning rate scheduler
        scheduler = self.surrogate.scheduler

        # Variables to track the best losses and weights
        best_meta_train_loss = np.inf
        best_meta_valid_loss = np.inf
        weights = copy.deepcopy(self.surrogate.state_dict())

        # Logging setup and initial information
        self.logger.debug(f"Starting meta-training for {num_epochs} epochs")
        self.logger.debug(f"Current learning rate: {scheduler.param_groups[0]['lr']:.5f}")
        self.logger.debug(f"Each dataset is sampled {num_inner_epochs} time(s)")

        # Main loop for meta-training + meta-validation
        for epoch in range(num_epochs):
            start_time = time.time()

            meta_train_loss = self.surrogate.fit(
                dataset_name=None,  # Sample random dataset
                num_epochs=num_inner_epochs,
            )

            # Logging the time and loss for this epoch
            meta_train_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - Meta-train loss: {meta_train_loss:.5f}"
            )
            self.logger.debug(
                f"Epoch {epoch+1}/{num_epochs} - Time: {meta_train_time:.2f}s"
            )

            # Meta-validation phase
            if epoch % valid_frequency == 0:
                # Perform meta-validation, which is just eval pass on the meta-valid split,
                # using all datasets in the split
                meta_valid_losses = []
                for dataset_name in self.sampler.metadataset.meta_splits["meta-valid"]:
                    pipeline_hps, metric_per_pipeline, metric = self.sampler.sample(
                        dataset_name=dataset_name,
                        num_samples=1,
                    )
                    meta_valid_loss = self.surrogate.validate(
                        pipeline_hps=pipeline_hps,
                        metric_per_pipeline=metric_per_pipeline,
                        metric=metric,
                    )

                    meta_valid_losses.append(meta_valid_loss)

                # Calculate the average loss across all datasets
                meta_valid_loss = np.mean(meta_valid_losses)

                # Logging meta-validation information
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - Meta-valid loss: {meta_valid_loss:.5f}"
                )
                self.logger.debug(
                    f"Epoch {epoch+1}/{num_epochs} - Time: {time.time() - meta_train_time:.2f}s"
                )

                # Check for improvement and update best weights if needed
                if meta_valid_loss < best_meta_valid_loss:
                    best_meta_valid_loss = meta_valid_loss
                    weights = copy.deepcopy(self.surrogate.state_dict())

                    self.logger.debug(
                        f"Epoch {epoch+1}/{num_epochs} - New best meta-valid loss: {best_meta_valid_loss:.5f}"
                    )
                    self.logger.debug(f"Epoch {epoch+1}/{num_epochs} - Saving weights...")

        # Load the best model weights and save them to a checkpoint file
        self.surrogate.load_state_dict(weights)
        self.surrogate.save_checkpoint(checkpoint_name="surrogate.pth")

    def run(
        self,
        max_patience: int = 16,
        loss_tol: float = 0.0001,
        meta_numn_epochs: int = 10,
        meta_num_inner_epochs: int = 1,
        meta_valid_frequency: int = 100,
    ):
        # Meta-train the surrogate model if num_epochs > 0,
        # otherwise load the checkpoint if exists

        self.meta_train_surrogate(
            num_epochs=meta_numn_epochs,
            num_inner_epochs=meta_num_inner_epochs,
            max_patience=max_patience,
            loss_tol=loss_tol,
            valid_frequency=meta_valid_frequency,
        )
        # self.acquisition.set_state(surrogate_model=self.surrogate)

        # TODO: generate candidates with another sampler
        # use acquisition to select candidate
