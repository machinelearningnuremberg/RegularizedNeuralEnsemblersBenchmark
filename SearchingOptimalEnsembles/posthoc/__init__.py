from typing import Callable

from .greedy_ensembler import GreedyEnsembler
from .neural_ensembler import NeuralEnsembler
from .quick_greedy_ensembler import QuickGreedyEnsembler
from .random_ensembler import RandomEnsembler
from .sk_stacker import ScikitLearnStacker
from .top_m_ensembler import TopMEnsembler
from .identity import IdentityEnsembler
from .temp_neural_ensembler import NeuralEnsembler as TestNeuralEnsembler

try:
    from .cmaes_ensembler import CMAESEnsembler
except Exception as e:
    print(f"Error importing CMAESEnsembler: {e}")
    CMAESEnsembler = None
try:
    from .des_ensembler import DESEnsembler
except Exception as e:
    print(f"Error importing DESEnsembler: {e}")
    DESEnsembler = None
from .single_best import SingleBest

EnsemblerMapping: dict[str, Callable] = {
    "random": RandomEnsembler,
    "greedy": GreedyEnsembler,
    "quick": QuickGreedyEnsembler,
    "topm": TopMEnsembler,
    "neural": NeuralEnsembler,
    "cmaes": CMAESEnsembler,
    "single": SingleBest,
    "des": DESEnsembler,
    "sks": ScikitLearnStacker,
    "identity": IdentityEnsembler,
    "test_neural": TestNeuralEnsembler
}
