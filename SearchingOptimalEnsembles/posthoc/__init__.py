from typing import Callable

# from .greedy_ensembler import GreedyEnsembler
from .random_ensembler import RandomEnsembler

EnsemblerMapping: dict[str, Callable] = {
    "random": RandomEnsembler,
    # "greedy": GreedyEnsembler,
}
