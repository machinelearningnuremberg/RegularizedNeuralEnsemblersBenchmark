from __future__ import annotations

from typing import Callable

from .ei import ExpectedImprovement
from .ucb import UpperConfidenceBound

AcquisitionMapping: dict[str, Callable] = {
    "ei": ExpectedImprovement,
    "ucb": UpperConfidenceBound,
}
