from __future__ import annotations

import numpy as np

from ...metadatasets.base_metadataset import BaseMetaDataset
from ...samplers import SamplerMapping
from ...utils.common import instance_from_map
from ..base_searcher import BaseSearcher


class RandomSearch(BaseSearcher):
    """Random search class."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        patience: int = 50,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(metadataset=metadataset, patience=patience)

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
            "random",
            name="sampler",
            kwargs=sampler_args,
        )

        self.logger.debug("Initialized Random search")

    def suggest(
        self,
        max_num_pipelines: int = 1,
        # batch_size: int,
        **kwargs,  # pylint: disable=unused-argument
    ) -> tuple[list, float]:
        # TODO: check on batch size

        num_pipelines = np.random.randint(1, max_num_pipelines + 1)
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
