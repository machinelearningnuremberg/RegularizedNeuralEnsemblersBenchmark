from __future__ import annotations

from typing_extensions import Literal

from .metadatasets import MetaDatasetMapping
from .searchers import SearcherMapping
from .utils.common import instance_from_map


def run(
    metadataset_name: Literal["scikit-learn", "nasbench201", "quicktune"],
    searcher_name: Literal["random", "bo"] = "bo",
    surrogate_name: Literal["dkl", "dre"] = "dkl",
    sampler_name: Literal["random"] = "random",
    acquisition_name: Literal["ei"] = "ei",
) -> None:
    """Runs the optimizer on the metadataset.

    Args:
        metadataset_name
        searcher_name
        surrogate_name
        sampler_name
        acquisition_name
    """

    metadataset = instance_from_map(
        MetaDatasetMapping,
        metadataset_name,
        name="metadataset",
    )

    searcher_args = {
        "metadataset": metadataset,
        "surrogate_name": surrogate_name,
        "sampler_name": sampler_name,
        "acquisition_name": acquisition_name,
        "initial_design_size": 5,
        "patience": 50,
    }

    searcher = instance_from_map(
        SearcherMapping,
        searcher_name,
        name="optimizer",
        kwargs=searcher_args,
    )

    searcher.run()
