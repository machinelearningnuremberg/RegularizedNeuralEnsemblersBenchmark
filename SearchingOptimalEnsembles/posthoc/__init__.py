from typing import Callable

from .greedy_ensembler import GreedyEnsembler
from .neural_ensembler import NeuralEnsembler
from .random_ensembler import RandomEnsembler
from .cmaes_ensembler import CMAESEnsembler
from .single_best import SingleBest


EnsemblerMapping: dict[str, Callable] = {
    "random": RandomEnsembler,
    "greedy": GreedyEnsembler,
    "neural": NeuralEnsembler,
    "cmaes": CMAESEnsembler,
    "single": SingleBest
}
