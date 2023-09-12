from __future__ import annotations

import numpy as np
import torch
import wandb

from ...metadatasets.base_metadataset import BaseMetaDataset
from ...samplers import SamplerMapping
from ...utils.common import instance_from_map
from ..base_searcher import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """ "Random search class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        worker_dir: str,
        patience: int = 50,
        initial_design_size: int = 5,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            metadataset=metadataset, worker_dir=worker_dir, patience=patience
        )

        sampler_args = {
            "metadataset": self.metadataset,
            "patience": self.patience,
            "device": self.device,
        }
        self.sampler = instance_from_map(
            SamplerMapping,
            "random",
            name="sampler",
            kwargs=sampler_args,
        )

        self.initial_design_size = initial_design_size

        self.logger.debug("Initialized Random search")

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

    def suggest(
        self,
    ):
        num_pipelines = np.random.randint(1, self.max_num_pipelines + 1)
        # Sample candidates

        ensembles_from_pending = self.sampler.generate_ensembles(
            candidates=self.X_pending,
            num_pipelines=1,
            batch_size=1,
        )

        if num_pipelines > 1:
            ensembles_from_observed = self.sampler.generate_ensembles(
                candidates=self.X_obs,
                num_pipelines=num_pipelines - 1,
                batch_size=1,
            )
            suggested_ensemble = np.concatenate(
                (ensembles_from_observed, ensembles_from_pending), axis=1
            ).tolist()[0]
        else:
            suggested_ensemble = ensembles_from_pending[0]

        return suggested_ensemble, ensembles_from_pending[0][0]

    def run(
        self,
        num_iterations: int = 100,
        max_num_pipelines: int = 1,
        dataset_id: int = 0,
        num_suggestions_per_batch: int = 1000,
        num_suggestion_batches: int = 5,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        # Set sampler, i.e. meta-test to random dataset
        dataset_name = self.metadataset.meta_splits["meta-test"][dataset_id]
        if wandb.run is not None:
            wandb.run.tags += (f"dataset={dataset_name}",)
        self.sampler.set_state(dataset_name=dataset_name, meta_split="meta-test")

        num_pipelines = 1
        # pylint: disable=unused-variable
        (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            time_per_pipeline,
            ensembles,
        ) = self.sampler.sample(
            max_num_pipelines=num_pipelines,
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

        for iteration in range(num_iterations):
            # pylint: disable=unused-variable

            # Evaluate candidates
            suggested_ensemble, suggested_pipeline = self.suggest()
            _, observed_metric, _, _ = self.metadataset.evaluate_ensembles(
                [suggested_ensemble]
            )

            post_hoc_ensemble, post_hoc_ensemble_metric = self.post_hoc_ensemble(
                num_suggestion_batches, num_suggestions_per_batch
            )
            if post_hoc_ensemble_metric < observed_metric:
                suggested_ensemble = post_hoc_ensemble
                observed_metric = post_hoc_ensemble_metric

            self.X_obs = np.concatenate((self.X_obs, [suggested_pipeline]))
            self.X_pending = np.setdiff1d(self.X_pending, [suggested_pipeline])

            X_pending = np.setdiff1d(X_pending, [suggested_pipeline])

            if observed_metric < self.incumbent:
                self.incumbent = observed_metric.item()
                self.incumbent_ensemble = suggested_ensemble
                self.logger.info(
                    f"Iteration {iteration + 1}/{num_iterations} - New incumbent: {self.incumbent:.5f}"
                )

            if wandb.run is not None:
                wandb.log({"searcher_iteration": iteration, "incumbent": self.incumbent})
                wandb.log(
                    {
                        "searcher_iteration": iteration,
                        "incumbent (norm)": self.compute_normalized_score(
                            torch.tensor(self.incumbent)
                        ),
                    }
                )

            # Increase the number of pipelines to sample if they are not exceeding the maximum
            # if num_pipelines < max_num_pipelines:
            #     num_pipelines += 1
            #     self.logger.debug(
            #         f"Increasing ensemble size to {num_pipelines} pipelines"
            #     )

            if X_pending.size == 0:
                self.logger.debug("No more pending pipelines. Stopping early...")
                break
