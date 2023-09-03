from __future__ import annotations

from typing_extensions import Literal

from .metadatasets import MetaDatasetMapping
from .searchers import SearcherMapping
from .utils.common import instance_from_map


def run(
    worker_dir: str,
    metadataset_name: Literal["scikit-learn", "nasbench201", "quicktune"],
    searcher_name: Literal["random", "bo"] = "bo",
    surrogate_name: Literal["dkl", "dre"] = "dkl",
    sampler_name: Literal["random"] = "random",
    acquisition_name: Literal["ei"] = "ei",
) -> None:
    """Runs the optimizer on the metadataset.

    Args:
        worker_dir: Directory where the results are stored.
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
        "worker_dir": worker_dir,
        "surrogate_name": surrogate_name,
        "sampler_name": sampler_name,
        "acquisition_name": acquisition_name,
        "initial_design_size": 5,
        "patience": 500,
    }

    searcher = instance_from_map(
        SearcherMapping,
        searcher_name,
        name="optimizer",
        kwargs=searcher_args,
    )

    run_args = {
        "loss_tolerance": 1e-4,
        "batch_size": 16,
        "meta_num_epochs": 50,
        "meta_num_inner_epochs": 1,
        "meta_valid_frequency": 100,
        "num_iterations": 1000,
        "num_inner_epochs": 1,
        "max_num_pipelines": 10,
    }

    searcher.run(**run_args)
