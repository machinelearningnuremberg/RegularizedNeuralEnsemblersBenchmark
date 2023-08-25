from __future__ import annotations

from typing import Callable

from .random_sampler import RandomSampler

SamplerMapping: dict[str, Callable] = {
    "random": RandomSampler,
}
