from __future__ import annotations

import numpy as np
from scipy.stats import norm as norm
from typing_extensions import Literal

from ...metadatasets.base_metadataset import BaseMetaDataset
from ...samplers import SamplerMapping
from ...utils.common import instance_from_map
from ..base_searcher import BaseSearcher
from ..simple_surrogates import create_surrogate


class LocalEnsembleOptimization(BaseSearcher):
    """Implements Bayesian Optimization for Ensemble Learning:  https://arxiv.org/abs/1605.06394"""
    def __init__(
        self,
        metadataset: BaseMetaDataset,
        patience: int = 50,
        surrogate_name: Literal["rf", "gp"] = "rf",
        surrogate_args: dict | None = None,
        acquisition_args: dict | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(metadataset=metadataset, patience=patience)

        assert surrogate_name in ["rf", "gp", "lightgbm"]

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

        if surrogate_args is None:
            surrogate_args = {}
        if acquisition_args is None:
            acquisition_args = {}

        self.logger.debug("Initialized LocalEnsembleOptimization searcher.")
        self.ensemble: list[int] = []
        self.len_first_X_obs = 0
        self.surrogate_args = surrogate_args
        self.surrogate_name = surrogate_name
        self.beta = acquisition_args.get("beta", 0.0)
        self._build_surrogate(**surrogate_args)

    def _build_surrogate(self, **kwargs):
        """Builds the surrogate model.
        Args:
            - surrogate_type: str, the type of surrogate model to use.
        """
        if self.surrogate_name == "gp":
            self.surrogate = create_surrogate("gp")
        elif self.surrogate_name == "rf":
            self.surrogate = create_surrogate(
                "rf", n_estimators=kwargs.get("num_estimators", 100)
            )
        elif self.surrogate_name == "lightgbm":
            self.surrogate = create_surrogate(
                "lightgbm", n_estimators=kwargs.get("num_estimators", 100), verbose=-1
            )
        else:
            raise NotImplementedError

    def _get_observations(self) -> tuple[np.array, np.array]:
        ensemble_ids = np.array(self.ensemble).reshape(1, -1).repeat(len(self.X_obs), 0)
        observed_ids = np.array(self.X_obs).reshape(-1, 1)
        ensembles = np.concatenate((ensemble_ids, observed_ids), axis=1).tolist()
        pipeline_hps, y, _, _ = self.metadataset.evaluate_ensembles(ensembles)
        X = pipeline_hps[:, -1, :].cpu().numpy()
        y = y.cpu().numpy()

        return X, y

    def EI(self, mean, sigma, incumbent):
        with np.errstate(divide="warn"):
            imp = incumbent - mean - self.beta
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _get_next_pipeline_to_observe(self) -> np.array:
        """Returns the next pipeline to observe.
        The candidate is selected from the observed pipelines.
        Returns: int, the id of the next pipeline to observe."""

        hp_pipelines, _, _, _ = self.metadataset.evaluate_ensembles(
            self.X_pending.reshape(-1, 1).tolist()
        )
        hp_pipelines = np.squeeze(hp_pipelines, axis=1)
        mean, sigma = self.surrogate.predict(hp_pipelines)
        acq_value = -self.EI(mean, sigma, self.incumbent)

        return self.X_pending[np.argmin(acq_value)]

    def _get_next_pipeline_to_add(self) -> tuple[int, float]:
        """Returns the next pipeline to add to the ensemble.
        The candidate is selected from the observed pipelines."""

        ensemble_ids = np.array(self.ensemble).reshape(1, -1).repeat(len(self.X_obs), 0)
        observed_ids = np.array(self.X_obs).reshape(-1, 1)
        ensembles = np.concatenate((ensemble_ids, observed_ids), axis=1).tolist()
        _, y, _, _ = self.metadataset.evaluate_ensembles(ensembles)
        y = y.cpu().numpy()
        argmin_set = np.where(y == np.min(y))[0]
        np.random.shuffle(argmin_set)
        pipeline_id = self.X_obs[argmin_set[0]]
        return pipeline_id, y[argmin_set[0]].item()

    def suggest(
        self,
        max_num_pipelines: int = 1,
        # batch_size: int,
        **kwargs,  # pylint: disable=unused-argument
    ) -> tuple[list[int], float]:
        # TODO: check on batch size

        if len(self.ensemble) == 0:
            self.ensemble = self.incumbent_ensemble
            self.len_first_X_obs = len(self.X_obs)
            i = 0
        else:
            i = len(self.X_obs) - self.len_first_X_obs

        j = i % max_num_pipelines
        self.ensemble.pop(j)
        X, y = self._get_observations()

        self.surrogate.fit(X, y)
        next_to_observe = self._get_next_pipeline_to_observe()

        next_to_add, _ = self._get_next_pipeline_to_add()
        self.ensemble.insert(j, next_to_add)

        return self.ensemble.copy(), next_to_observe
