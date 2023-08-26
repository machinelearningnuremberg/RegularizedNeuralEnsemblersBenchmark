from __future__ import annotations

from typing import Callable

from .dkl import DeepKernelGP

ModelMapping: dict[str, Callable] = {
    "dkl": DeepKernelGP,
}
