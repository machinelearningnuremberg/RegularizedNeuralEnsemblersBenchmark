from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from scipy.stats import norm as norm
import torch
import wandb
from scipy.stats import norm as norm
from scipy.stats import rankdata
from typing_extensions import Literal

from ...metadatasets.base_metadataset import BaseMetaDataset
from ...posthoc.greedy_ensembler import GreedyEnsembler
from ...samplers import SamplerMapping
from ...samplers.diversity_sampler import DiversitySampler
from ...utils.common import instance_from_map
from ..base_searcher import BaseSearcher
from ..bayesian_optimization.acquisition.ei import ExpectedImprovement
from ..simple_surrogates import create_surrogate


class DivBO(BaseSearcher):
    """Diversity-aware Bayesian Optimization proposed in https://arxiv.org/pdf/2302.03255.pdf"""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        patience: int = 50,
        pool_size: int = 25,
        surrogate_name: Literal["rf", "gp"] = "rf",
        surrogate_args: dict | None = None,
        acquisition_args: dict | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(metadataset=metadataset, patience=patience)

        assert surrogate_name in ["rf", "gp"]

        diversity_surrogate_name = "lightgbm"
        diversity_surrogate_args = None

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

        self.diversity_sampler = DiversitySampler(
            metadataset=self.metadataset,
            patience=self.patience,
            device=self.device,
        )

        if surrogate_args is None:
            surrogate_args = {}

        if diversity_surrogate_args is None:
            diversity_surrogate_args = {}

        if acquisition_args is None:
            acquisition_args = {}

        self.logger.debug("Initialized LocalEnsembleOptimization searcher.")

        self.pool_size = pool_size
        self.posthoc_ensembler = GreedyEnsembler(
            metadataset=self.metadataset, device=self.device
        )
        self.performance_surrogate_name = surrogate_name
        self.diversity_surrogate_name = diversity_surrogate_name

        self.performance_surrogate = self._build_surrogate(
            surrogate_name, **surrogate_args
        )
        self.diversity_surrogate = self._build_surrogate(
            diversity_surrogate_name, **diversity_surrogate_args
        )
        self.beta = acquisition_args.get("beta", 0.0)
        self.psi = acquisition_args.get("psi", 0.05)  # beta in the original paper
        self.tau = acquisition_args.get("tau", 0.2)  # tau in the original paper
        self.num_samples = acquisition_args.get("num_samples", 15)
        self.observed_pipeline_ids: list[int] = []

    def _build_surrogate(self, surrogate_name, **kwargs):
        """Builds the surrogate model.
        Args:
            - surrogate_type: str, the type of surrogate model to use.
        """
        if surrogate_name == "gp":
            surrogate = create_surrogate("gp")
        elif surrogate_name == "rf":
            surrogate = create_surrogate(
                "rf", n_estimators=kwargs.get("n_estimators", 100)
            )
        elif surrogate_name == "lightgbm":
            surrogate = create_surrogate(
                "lightgbm", n_estimators=kwargs.get("n_estimators", 100), verbose=-1
            )
        else:
            raise NotImplementedError

        return surrogate

    def EI(self, mean, sigma, incumbent):
        with np.errstate(divide="warn"):
            imp = incumbent - mean - self.beta
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def get_weight(self):
        x = 1.0 / (1.0 + np.exp(-self.tau * self.iteration))
        return self.psi * (x - 0.5)

    def diversity_acquisition(self, X_pool, X_pending):
        div_matrix = np.zeros((len(X_pool), self.num_samples, len(X_pending)))
        for i in range(len(X_pool)):
            temp_X = np.concatenate(
                [np.tile(X_pool[i], (len(X_pending), 1)), X_pending], axis=1
            )
            mean, std = self.diversity_surrogate.predict(temp_X)
            X = np.random.normal(
                mean.reshape(1, -1),
                std.reshape(1, -1),
                (self.num_samples, len(X_pending)),
            )
            div_matrix[i] = X

        div_aqcf = -div_matrix.min(0).mean(0)
        return div_aqcf

    def suggest(
        self,
        max_num_pipelines: int = 1,
        **kwargs,  # pylint: disable=unused-argument
    ):
        # fit performance surrogate

        if len(self.X_obs) == 0:
            _, metric, _, _, ensembles = self.initial_design_sampler.sample(
                fixed_num_pipelines=max_num_pipelines,
                batch_size=1,
                observed_pipeline_ids=None,
            )

            suggested_pipeline = ensembles[0]
            suggested_ensemble = ensembles
        else:
            (
                pipeline_hps,
                metric,
                metric_per_pipeline,
                time_per_pipeline,
            ) = self.metadataset.evaluate_ensembles(ensembles=[self.X_obs.tolist()])

            y_perf = metric_per_pipeline.reshape(-1, 1)
            X_perf = pipeline_hps.cpu().numpy()[0]

            self.performance_surrogate.fit(X_perf, y_perf)

            x1, x2, score = self.diversity_sampler.sample_with_diversity_score(
                observed_pipeline_ids=self.X_obs
            )
            x1 = x1.cpu().numpy()
            x2 = x2.cpu().numpy()
            y_div = score.cpu().numpy()
            X_div = np.concatenate([x1, x2], axis=1)

            self.diversity_surrogate.fit(X_div, y_div)

            # post hoc ensembling
            if len(self.X_obs) > self.pool_size:
                pool, _ = self.posthoc_ensembler.sample(
                    X_obs=self.X_obs, max_num_pipelines=self.pool_size
                )

            else:
                pool = self.X_obs

            # acquisition function
            pipeline_hps, _, _, _ = self.metadataset.evaluate_ensembles(
                ensembles=[self.X_pending]
            )
            X_pending = pipeline_hps.cpu().numpy()[0]

            pipeline_hps, _, _, _ = self.metadataset.evaluate_ensembles(ensembles=[pool])
            X_pool = pipeline_hps.cpu().numpy()[0]

            mean, std = self.performance_surrogate.predict(X_pending)
            perf_acqf = -self.EI(mean, std, self.incumbent)

            div_acqf = self.diversity_acquisition(X_pool=X_pool, X_pending=X_pending)

            acqf = rankdata(perf_acqf) + self.get_weight() * rankdata(div_acqf)

            suggested_pipeline = self.X_pending[np.argmin(acqf)]
            suggested_ensemble = pool[:max_num_pipelines]

        return suggested_ensemble, suggested_pipeline
