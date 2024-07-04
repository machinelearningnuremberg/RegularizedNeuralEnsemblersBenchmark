from typing import Callable

from .greedy_ensembler import GreedyEnsembler
from .neural_ensembler import NeuralEnsembler
from .random_ensembler import RandomEnsembler
from .sk_stacker import ScikitLearnStacker

try:
    from .cmaes_ensembler import CMAESEnsembler
except Exception as e:
    print(f"Error importing CMAESEnsembler: {e}")
    CMAESEnsembler = None
from .single_best import SingleBest
from .des_ensembler import DESEnsembler

EnsemblerMapping: dict[str, Callable] = {
    "random": RandomEnsembler,
    "greedy": GreedyEnsembler,
    "neural": NeuralEnsembler,
    "cmaes": CMAESEnsembler,
    "single": SingleBest,
    "des": DESEnsembler,
    "sks": ScikitLearnStacker
}
