from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..utils.common import move_to_device
from ..utils.logger import get_logger


class BaseSampler:
    def __init__(
        self,
        metadataset: BaseMetaDataset,
        patience: int = 50,
        device: torch.device = torch.device("cpu"),
    ):
        """Base class for samplers. The sampler is used to sample a batch of pipelines
        from the hyperparameter candidates of a dataset.

        Args:
            metadataset (BaseMetaDataset): Meta-dataset.
            patience (int, optional): Patience. Defaults to 50.
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
        """
        if patience < 1:
            raise ValueError("Patience should be at least 1")

        self.metadataset = metadataset
        self.patience = patience
        self.device = device
        self.logger = get_logger(name="SEO-SAMPLER", logging_level="debug")

    def generate_ensembles(
        self, candidates: np.ndarray, num_pipelines: int = 10, batch_size: int = 16
    ) -> list[list[int]]:
        """Generate ensembles of pipelines.

        Args:
            candidates (np.ndarray): Pipeline IDS of the candidates.
            num_pipelines (int): Number of pipelines in the ensemble.
            batch_size (int): Batch size.

        Returns:
            list[list[int]]: List of ensembles.
        """

        raise NotImplementedError

    @move_to_device
    def sample(
        self,
        max_num_pipelines: int = 10,
        fixed_num_pipelines: int | None = None,
        batch_size: int = 16,
        observed_pipeline_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
        """Sample a batch of pipelines along with metrics. The pipelines are sampled from the
        hyperparameter candidates of the dataset specified in self.dataset_name.

        Args:
            max_num_pipelines (int, optional): Maximum number of pipelines in the
                ensemble. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 16.
            observed_pipeline_ids (list[int], optional): List of observed pipeline IDs.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
                Tuple containing the hyperparameters, metric, metric per pipeline, time per
                pipeline, and the ensembles as list of lists.
        """

        if observed_pipeline_ids is None or len(observed_pipeline_ids) == 0:
            candidates = self.metadataset.hp_candidates_ids.numpy().astype(int)
        else:
            candidates = np.array(observed_pipeline_ids)

        if fixed_num_pipelines is not None:
            num_pipelines = fixed_num_pipelines
        else:
            num_pipelines = np.random.randint(1, max_num_pipelines + 1)

        ensembles = self.generate_ensembles(candidates, num_pipelines, batch_size)

        (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            time_per_pipeline,
        ) = self.metadataset.evaluate_ensembles(ensembles=ensembles)

        return pipeline_hps, metric, metric_per_pipeline, time_per_pipeline, ensembles

    def set_state(
        self, dataset_name: str | None = None, meta_split: str = "meta-train"
    ) -> None:
        """Set the state of the sampler. This includes populating the dataset name.
        If dataset_name is None, a random dataset is chosen from the specified
        meta-split.

        Args:
            dataset_name (str, optional): Name of the dataset. Defaults to None.
            meta_split (str, optional): Meta-split name. Defaults to "meta-train".
        """

        if dataset_name is None:
            dataset_name = np.random.choice(self.metadataset.meta_splits[meta_split])

        self.metadataset.set_state(dataset_name=dataset_name)
