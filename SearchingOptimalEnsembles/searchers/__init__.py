from __future__ import annotations

from typing import Callable

from .bayesian_optimization.searcher import BayesianOptimization
from .local_ensemble_optimization.searcher import LocalEnsembleOptimization
from .random_search.searcher import RandomSearch

SearcherMapping: dict[str, Callable] = {
    "bo": BayesianOptimization,
    "random": RandomSearch,
    "leo": LocalEnsembleOptimization,
}
