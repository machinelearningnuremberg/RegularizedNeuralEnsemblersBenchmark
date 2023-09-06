from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from typing_extensions import Literal

from ...metadatasets.base_metadataset import BaseMetaDataset
from ...samplers import SamplerMapping
from ...utils.common import instance_from_map
from ..base_searcher import BaseOptimizer
from .acquisition import AcquisitionMapping
from .models import ModelMapping


class BayesianOptimization(BaseOptimizer):
    """Bayesian optimization class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        worker_dir: str,
        patience: int = 50,
        surrogate_name: Literal["dkl", "dre"] = "dkl",
        surrogate_args: dict | None = None,
        acquisition_args: dict | None = None,
        sampler_name: Literal["random", "local_search"] = "random",
        acquisition_name: Literal["ei", "ucb"] = "ei",
        initial_design_size: int = 5,
        checkpoint_path: str | None = None,
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
        super().__init__(
            metadataset=metadataset, worker_dir=worker_dir, patience=patience
        )

        if checkpoint_path is None:
            self.checkpoint_path = Path(worker_dir) / "checkpoints"
        else:
            self.checkpoint_path = Path(checkpoint_path)

        sampler_args = {
            "metadataset": self.metadataset,
            "patience": self.patience,
            "device": self.device,
        }
        self.initial_design_sampler = instance_from_map(
            SamplerMapping,
            "random",
            name="initial_design_sampler",
            kwargs=sampler_args,
        )
        self.sampler = instance_from_map(
            SamplerMapping,
            sampler_name,
            name="sampler",
            kwargs=sampler_args,
        )

        default_args = ModelMapping[surrogate_name].default_config.copy()
        surrogate_args = {**default_args, **(surrogate_args or {})}
        surrogate_args.update(
            {
                "sampler": self.sampler,
                "checkpoint_path": self.checkpoint_path,
                "device": self.device,
            }
        )
        self.surrogate = instance_from_map(
            ModelMapping,
            surrogate_name,
            name="surrogate",
            kwargs=surrogate_args,
        )

        acquisition_args = {**(acquisition_args or {})}
        acquisition_args.update(
            {
                "device": self.device,
            }
        )
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

    def meta_train_surrogate(
        self,
        num_epochs: int = 10000,
        num_inner_epochs: int = 1,
        loss_tol: float = 0.0001,
        valid_frequency: int = 100,
        max_num_pipelines: int = 10,
        batch_size: int = 16,
    ) -> None:
        """
        Perform meta-training on the surrogate model.

        This training includes both a meta-train and meta-validation phase to adjust the model's
        weights for generalized learning.

        Args:
            num_epochs (int, optional): Total number of meta-training epochs. Defaults to 100.
            num_inner_epochs (int, optional): Number of epochs for each inner optimization loop.
                These inner epochs train on individual datasets. Defaults to 100.
            loss_tol (float, optional): Tolerance for loss improvement. Considered as no improvement
                if less than this value. Defaults to 0.0001.
            valid_frequency (int, optional): Frequency of performing meta-validation, in terms of epochs.
                Defaults to 100000.
            max_num_pipelines (int, optional): Maximum number of pipelines to sample. Defaults to 10.
            batch_size (int, optional): Batch size for each inner optimization loop. Defaults to 16.

        Returns:
            None: Updates the model's state in-place and saves the best state to a checkpoint file.
        """

        # Initialize the surrogate model
        self.surrogate.checkpointer.load_checkpoint(
            checkpoint_name=f"{self.surrogate.__class__.__name__}_{self.metadataset.__class__.__name__}.pth"
        )

        # Initialize the learning rate optimizer
        optimizer = self.surrogate.optimizer

        # Variables to track the best losses and weights
        meta_valid_losses = [np.inf]
        best_meta_valid_loss = np.inf
        weights = copy.deepcopy(self.surrogate.state_dict())

        # Logging setup and initial information
        self.logger.debug(f"Starting meta-training for {num_epochs} epochs")
        self.logger.debug(f"Current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        self.logger.debug(f"Each dataset is sampled {num_inner_epochs} time(s)")

        patience = self.patience
        # Main loop for meta-training + meta-validation
        for epoch in range(num_epochs):
            if patience <= 0:  # Check for patience
                self.logger.debug("Patience reached zero. Stopping early...")
                break

            # Set sampler, i.e. meta-train to random dataset
            # self.sampler.set_state(dataset_name="kr-vs-kp", meta_split="meta-train")
            start_time = time.time()
            self.sampler.set_state(dataset_name=None, meta_split="meta-train")
            sampler_time = time.time() - start_time

            start_time = time.time()
            meta_train_loss = self.surrogate.fit(
                num_epochs=num_inner_epochs,
                max_num_pipelines=max_num_pipelines,
                batch_size=batch_size,
            )

            if meta_train_loss is None:
                self.logger.info("Encountered NaN loss. Reducing patience...")
                patience -= 1  # Reduce patience
                continue

            # Logging the time and loss for this epoch
            meta_train_time = time.time() - start_time
            self.logger.debug(
                f"Epoch {epoch+1}/{num_epochs} - Meta-train - Loss: {meta_train_loss:.5f} - Time: load: {sampler_time:.2f}s, pass {meta_train_time:.2f}s"
            )
            if wandb.run is not None:
                wandb.log({"epoch": epoch, "meta_train_loss": meta_train_loss})

            # Meta-validation phase
            if (epoch + 1) % valid_frequency == 0:
                # Perform meta-validation, which is just eval pass on the meta-valid split,
                # using all datasets in the split
                start_time = time.time()
                _meta_valid_losses = []
                for dataset_name in self.sampler.metadataset.meta_splits["meta-valid"]:
                    self.sampler.set_state(
                        dataset_name=dataset_name, meta_split="meta-valid"
                    )
                    # pylint: disable=unused-variable
                    (
                        pipeline_hps,
                        metric,
                        metric_per_pipeline,
                        time_per_pipeline,
                        ensembles,
                    ) = self.sampler.sample(
                        max_num_pipelines=max_num_pipelines, batch_size=batch_size
                    )
                    meta_valid_loss = self.surrogate.validate(
                        pipeline_hps=pipeline_hps,
                        metric_per_pipeline=metric_per_pipeline,
                        metric=metric,
                    )

                    _meta_valid_losses.append(meta_valid_loss)

                # Calculate the average loss across all datasets
                meta_valid_loss = np.mean([loss.item() for loss in _meta_valid_losses])
                if wandb.run is not None:
                    wandb.log({"epoch": epoch, "meta_valid_loss": meta_valid_loss})
                meta_valid_losses.append(meta_valid_loss)

                # Logging meta-validation information
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - Meta-valid - Loss: {meta_valid_loss:.5f} - Time: {time.time() - start_time:.2f}s"
                )

                # Check for improvement and update best weights if needed
                if meta_valid_loss < best_meta_valid_loss:
                    best_meta_valid_loss = meta_valid_loss
                    weights = copy.deepcopy(self.surrogate.state_dict())

                    self.logger.debug(
                        f"Epoch {epoch+1}/{num_epochs} - New best meta-valid loss: {best_meta_valid_loss:.5f}"
                    )
                    self.logger.debug(f"Epoch {epoch+1}/{num_epochs} - Saving weights...")

                # Check for early stopping
                if meta_valid_losses[-1] > meta_valid_losses[-2] + loss_tol:
                    self.logger.debug(
                        f"Epoch {epoch+1}/{num_epochs} - No improvement in meta-valid loss. Reducing patience..."
                    )
                    patience -= 1  # Reduce patience

        # Load the best model weights and save them to a checkpoint file
        self.surrogate.load_state_dict(weights)
        self.surrogate.checkpointer.save_checkpoint(
            checkpoint_name=f"{self.surrogate.__class__.__name__}_{self.metadataset.__class__.__name__}.pth"
        )

    def post_hoc_ensemble(
        self, num_batches: int = 5, num_suggestions_per_batch: int = 1000
    ):
        best_score = np.inf
        best_ensemble = None
        for iterations in range(num_batches):
            num_pipelines = np.random.randint(1, self.max_num_pipelines + 1)
            ensembles = self.sampler.generate_ensembles(
                candidates=self.X_obs,
                num_pipelines=num_pipelines,
                batch_size=num_suggestions_per_batch,
            )

            _, metric, _, _ = self.metadataset.evaluate_ensembles(ensembles)
            temp_best_metric = metric.min()
            temp_best_id = metric.argmin()

            if temp_best_metric < best_score:
                best_score = temp_best_metric
                best_ensemble = ensembles[temp_best_id]

        return best_ensemble, best_score

    def suggest(self, num_batches: int = 5, num_suggestions_per_batch: int = 1000):
        suggested_ensemble = None
        suggested_pipeline = None
        best_score = np.inf

        for iterations in range(num_batches):
            num_pipelines = np.random.randint(1, self.max_num_pipelines + 1)

            if num_pipelines > 1:
                # Sample candidates
                ensembles_from_observed = self.sampler.generate_ensembles(
                    candidates=self.X_obs,
                    num_pipelines=num_pipelines - 1,
                    batch_size=num_suggestions_per_batch,
                )
                (
                    pipeline_hps,
                    _,
                    metric_per_pipeline,
                    _,
                ) = self.metadataset.evaluate_ensembles(ensembles_from_observed)

            ensembles_from_pending = self.sampler.generate_ensembles(
                candidates=self.X_pending,
                num_pipelines=1,
                batch_size=num_suggestions_per_batch,
            )
            new_pipeline_hps, _, _, _ = self.metadataset.evaluate_ensembles(
                ensembles_from_pending
            )

            if num_pipelines > 1:
                # Combine the observed and pending pipelines
                query_pipeline_hps = torch.cat(
                    (pipeline_hps, new_pipeline_hps), dim=1
                ).to(self.device)

                # for DRE, ideally should be in DRE
                new_metric_per_pipeline = torch.zeros(len(new_pipeline_hps), 1)
                metric_per_pipeline = torch.cat(
                    (metric_per_pipeline, new_metric_per_pipeline), dim=1
                ).to(self.device)
            else:
                # TODO: FIX add metric_per_pipeline for DRE

                query_pipeline_hps = new_pipeline_hps.to(self.device)
                metric_per_pipeline = torch.zeros(len(new_pipeline_hps), 1).to(
                    self.device
                )

            # Evaluate the acquisition function
            score = self.acquisition.eval(
                x=query_pipeline_hps,
                metric_per_pipeline=metric_per_pipeline,
            )

            # Select the best candidate
            iter_best_score = torch.min(score)
            idx_condition = torch.where(score == iter_best_score)[0]
            iter_best_idx = idx_condition[torch.randint(len(idx_condition), (1,))]
            iter_best_score = score[iter_best_idx]

            if iter_best_score < best_score:
                best_score = iter_best_score

                suggested_ensemble = []
                if num_pipelines > 1:
                    suggested_ensemble = ensembles_from_observed[iter_best_idx]

                suggested_ensemble += ensembles_from_pending[iter_best_idx]
                suggested_pipeline = ensembles_from_pending[iter_best_idx][0]

        return suggested_ensemble, suggested_pipeline

    def run(
        self,
        loss_tolerance: float = 1e-4,
        batch_size: int = 16,
        meta_num_epochs: int = 50,
        meta_num_inner_epochs: int = 1,
        meta_valid_frequency: int = 100,
        num_iterations: int = 1000,
        num_inner_epochs: int = 1,
        max_num_pipelines: int = 1,
    ) -> None:
        # Meta-train the surrogate model if num_epochs > 0,
        # otherwise load the checkpoint if exists

        self.batch_size = batch_size
        self.meta_train_surrogate(
            num_epochs=meta_num_epochs,
            num_inner_epochs=meta_num_inner_epochs,
            loss_tol=loss_tolerance,
            valid_frequency=meta_valid_frequency,
            max_num_pipelines=max_num_pipelines,
            batch_size=batch_size,
        )

        # Set sampler, i.e. meta-test to random dataset
        self.sampler.set_state(dataset_name=None, meta_split="meta-test")

        # Sample initial design points
        self.logger.debug(f"Sampling {self.initial_design_size} initial design points")
        # pylint: disable=unused-variable
        (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            time_per_pipeline,
            ensembles,
        ) = self.initial_design_sampler.sample(
            fixed_num_pipelines=max_num_pipelines,
            batch_size=self.initial_design_size,
            observed_pipeline_ids=None,
        )

        # Bookkeeping variables
        self.X_obs = np.unique(ensembles)
        X_pending = np.array(self.metadataset.hp_candidates_ids)
        self.X_pending = np.setdiff1d(X_pending, self.X_obs)
        self.incumbent = torch.min(metric).item()
        self.incumbent_ensemble = ensembles[torch.argmin(metric).item()]
        self.num_pipelines = len(self.X_obs)
        self.max_num_pipelines = max_num_pipelines

        self.logger.debug(f"Number of pending pipelines: {len(X_pending)}")
        self.logger.info(f"Initial incumbent: {self.incumbent:.5f}")

        # Main loop for Bayesian optimization
        for iteration in range(num_iterations):
            # Fine-tune the surrogate model based on the observed data
            self.iteration = iteration
            self.surrogate.fit(
                num_epochs=num_inner_epochs,
                observed_pipeline_ids=self.X_obs,
                max_num_pipelines=max_num_pipelines,
                batch_size=batch_size,
            )

            self.acquisition.set_state(
                surrogate_model=self.surrogate, incumbent=self.incumbent
            )

            # Evaluate candidates
            suggested_ensemble, suggested_pipeline = self.suggest()
            _, observed_metric, _, _ = self.metadataset.evaluate_ensembles(
                [suggested_ensemble]
            )

            if max_num_pipelines > 1:
                post_hoc_ensemble, post_hoc_ensemble_metric = self.post_hoc_ensemble()
                if post_hoc_ensemble_metric < observed_metric:
                    suggested_ensemble = post_hoc_ensemble
                    observed_metric = post_hoc_ensemble_metric

            # Update bookkeeping variables
            self.X_obs = np.concatenate((self.X_obs, [suggested_pipeline]))
            self.X_pending = np.setdiff1d(self.X_pending, [suggested_pipeline])

            if observed_metric < self.incumbent:
                self.incumbent = observed_metric.item()
                self.incumbent_ensemble = suggested_ensemble
                self.logger.info(
                    f"Iteration {iteration + 1}/{num_iterations} - New incumbent: {self.incumbent:.5f}"
                )

            if wandb.run is not None:
                wandb.log({"iteration": iteration, "incumbent": self.incumbent})

            # Increase the number of pipelines to sample if they are not exceeding the maximum
            # if self.num_pipelines < max_num_pipelines:
            #    self.num_pipelines += 1
            #    self.logger.debug(
            #        f"Increasing ensemble size to {self.num_pipelines} pipelines"
            #    )

            if X_pending.size == 0:
                self.logger.debug("No more pending pipelines. Stopping early...")
                break
