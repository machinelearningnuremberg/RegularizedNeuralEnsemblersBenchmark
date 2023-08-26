from __future__ import annotations

from typing import Callable

from .bayesian_optimization.searcher import BayesianOptimization

SearcherMapping: dict[str, Callable] = {
    "bo": BayesianOptimization,
}
