from __future__ import annotations

from typing import Callable

from .dkl import DeepKernelGP
from .dre import DRE
from .rf import BootstrapRandomForest

ModelMapping: dict[str, Callable] = {
    "dkl": DeepKernelGP,
    "dre": DRE,
    "rf": BootstrapRandomForest,
}
