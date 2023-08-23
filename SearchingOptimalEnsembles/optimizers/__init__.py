from __future__ import annotations

from typing import Callable

from .bayesian_optimization.optimizer import BayesianOptimization

# from .bayesian_optimization.models import ModelMapping
# from .bayesian_optimization.acquisition import AcquisitionMapping

OptimizerMapping: dict[str, Callable] = {
    "bo": BayesianOptimization,
}
