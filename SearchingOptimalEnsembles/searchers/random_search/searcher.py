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

        self.logger.debug("Initialized Random search")

    def run(
        self,
        num_iterations: int = 100,
        max_num_pipelines: int = 1,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        # Set sampler, i.e. meta-test to random dataset
        self.sampler.set_state(dataset_name=None, meta_split="meta-test")

        num_pipelines = 1

        # Bookkeeping variables
        X_pending = np.array(self.metadataset.hp_candidates_ids)
        incumbent = np.inf

        for iteration in range(num_iterations):
            # pylint: disable=unused-variable
            (
                pipeline_hps,
                metric,
                metric_per_pipeline,
                time_per_pipeline,
                ensembles,
            ) = self.sampler.sample(
                max_num_pipelines=num_pipelines,
                batch_size=1,
                observed_pipeline_ids=None,
            )

            best_metric = torch.min(metric).item()

            if best_metric < incumbent:
                incumbent = best_metric
                self.logger.info(
                    f"Iteration {iteration+1}/{num_iterations} - New incumbent: {incumbent:.5f}"
                )

            if wandb.run is not None:
                wandb.log({"iteration": iteration, "incumbent": incumbent})

            # Increase the number of pipelines to sample if they are not exceeding the maximum
            if num_pipelines < max_num_pipelines:
                num_pipelines += 1
                self.logger.debug(
                    f"Increasing ensemble size to {num_pipelines} pipelines"
                )

            if X_pending.size == 0:
                self.logger.debug("No more pending pipelines. Stopping early...")
                break
