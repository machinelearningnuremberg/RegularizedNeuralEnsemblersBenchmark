from __future__ import annotations

from typing import Callable

from .ei import ExpectedImprovement
from .lcb import LowerConfidenceBound

AcquisitionMapping: dict[str, Callable] = {
    "ei": ExpectedImprovement,
    "lcb": LowerConfidenceBound,
}
