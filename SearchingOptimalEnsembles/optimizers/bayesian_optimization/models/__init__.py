from __future__ import annotations

from typing import Callable

from .deep_kernel_gp import DeepKernelGP

ModelMapping: dict[str, Callable] = {
    "dkl": DeepKernelGP,
}
