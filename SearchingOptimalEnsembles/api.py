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
    run_args: dict = None,
    meta_num_epochs: int = 0,
    max_num_pipelines: int = 1,
    dataset_id: int = 0,
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
        "surrogate_args": run_args.pop("surrogate_args", None),
        "acquisition_args": run_args.pop("acquisition_args", None),
        "initial_design_size": 5,
        "patience": 500,
    }

    searcher = instance_from_map(
        SearcherMapping,
        searcher_name,
        name="searcher",
        kwargs=searcher_args,
    )

    default_run_args = {
        "loss_tolerance": 1e-4,
        "batch_size": 16,
        "meta_num_epochs": meta_num_epochs,
        "meta_num_inner_epochs": 1,
        "meta_valid_frequency": 10,
        "num_iterations": 100,
        "num_inner_epochs": 1,
        "max_num_pipelines": max_num_pipelines,
        "dataset_id": dataset_id,
    }

    if run_args is None:
        run_args = default_run_args
    else:
        default_run_args.update(run_args)
        run_args = default_run_args

    print(run_args)
    searcher.run(**run_args)
