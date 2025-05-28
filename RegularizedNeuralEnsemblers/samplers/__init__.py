from __future__ import annotations

from typing import Callable

from .local_search_sampler import LocalSearchSampler
from .random_sampler import RandomSampler

SamplerMapping: dict[str, Callable] = {
    "random": RandomSampler,
    "local_search": LocalSearchSampler,
}
