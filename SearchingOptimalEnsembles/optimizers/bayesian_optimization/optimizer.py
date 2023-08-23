from __future__ import annotations

from pathlib import Path

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
        surrogate_name: Literal["dkl", "dre"] = "dkl",
        sampler_name: Literal["random"] = "random",
        acquisition_name: Literal["ei"] = "ei",
        initial_design_size: int = 5,
        patience: int = 50,
        logger=None,
        budget: None | int | float = None,
    ):
        super().__init__(
            metadataset=metadataset, patience=patience, logger=logger, budget=budget
        )

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
            "checkpoint_path": Path(
                "/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/checkpoints"
            ),
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

    def meta_train(self):
        pass

    def run(self):
        dataset_name = "kr-vs-kp"
        meta_split = "meta-train"

        self.sampler.set_state(dataset_name=dataset_name, meta_split=meta_split)
        self.surrogate.train(training_epochs=10)
        self.acquisition.set_state(surrogate_model=self.surrogate)

        # TODO: generate candidates with another sampler
        # use acquisition to select candidate
