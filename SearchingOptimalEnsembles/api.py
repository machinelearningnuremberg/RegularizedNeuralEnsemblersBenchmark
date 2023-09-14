from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import wandb
from typing_extensions import Literal

from .metadatasets import MetaDatasetMapping
from .posthoc import EnsemblerMapping
from .searchers import SearcherMapping
from .utils.common import instance_from_map


def run(
    worker_dir: str,
    metadataset_name: Literal["scikit-learn", "nasbench201", "quicktune"],
    #############################################
    searcher_name: Literal["random", "bo"] = "bo",
    initial_design_size: int = 5,
    #############################################
    surrogate_name: Literal["dkl", "dre"] = "dkl",
    surrogate_args: dict | None = None,
    checkpoint_path: str | None = None,
    #############################################
    acquisition_name: Literal["ei", "lcb"] = "ei",
    acquisition_args: dict | None = None,
    #############################################
    ensembler_name: Literal["greedy", "random"] = "random",
    sampler_name: Literal["random"] = "random",
    #############################################
    meta_num_epochs: int = 0,
    meta_num_inner_epochs: int = 1,
    meta_valid_frequency: int = 10,
    loss_tolerance: float = 1e-4,
    #############################################
    num_iterations: int = 100,
    num_suggestion_batches: int = 5,
    num_suggestions_per_batch: int = 1000,
    num_inner_epochs: int = 1,
    batch_size: int = 16,
    max_num_pipelines: int = 1,
    apply_posthoc_ensemble: bool = True,
    #############################################
    dataset_id: int = 0,
) -> None:
    """Runs SOE on the metadataset.

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

    if checkpoint_path is None:
        checkpoint_dir = Path(worker_dir) / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_path)
    # Create the checkpoint directory if it does not exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    searcher_args = {
        "metadataset": metadataset,
        "surrogate_name": surrogate_name,
        "surrogate_args": surrogate_args,
        "checkpoint_dir": checkpoint_dir,
        "acquisition_name": acquisition_name,
        "acquisition_args": acquisition_args,
        "sampler_name": sampler_name,
        "patience": 500,
    }
    searcher = instance_from_map(
        SearcherMapping,
        searcher_name,
        name="searcher",
        kwargs=searcher_args,
    )

    ensembler_args = {
        "metadataset": metadataset,
        "device": searcher.device,
    }
    posthoc_ensembler = instance_from_map(
        EnsemblerMapping,
        ensembler_name,
        name="posthoc_ensembler",
        kwargs=ensembler_args,
    )

    if meta_num_epochs > 0:
        assert hasattr(
            searcher, "surrogate"
        ), "Surrogate must be defined for meta-training."
        # TODO: assert that the surrogate can be meta_trained

        searcher.meta_train_surrogate(
            num_epochs=meta_num_epochs,
            num_inner_epochs=meta_num_inner_epochs,
            loss_tol=loss_tolerance,
            valid_frequency=meta_valid_frequency,
            max_num_pipelines=max_num_pipelines,
            batch_size=batch_size,
        )

    # Set sampler based on the dataset id
    dataset_name = metadataset.meta_splits["meta-test"][dataset_id]
    if wandb.run is not None:
        wandb.run.tags += (f"dataset={dataset_name}",)
    searcher.sampler.set_state(dataset_name=dataset_name, meta_split="meta-test")

    # Sample initial design
    _, metric, _, _, ensembles = searcher.initial_design_sampler.sample(
        fixed_num_pipelines=max_num_pipelines,
        batch_size=initial_design_size,
        observed_pipeline_ids=None,
    )

    # Bookkeeping variables
    X_obs = np.unique(ensembles)
    X_pending = np.array(metadataset.hp_candidates_ids)
    X_pending = np.setdiff1d(X_pending, X_obs)
    incumbent = torch.min(metric).item()
    incumbent_ensemble = ensembles[torch.argmin(metric).item()]

    # Main search loop
    for iteration in range(num_iterations):
        searcher.set_state(
            X_obs=X_obs,
            X_pending=X_pending,
            incumbent=incumbent,
            incumbent_ensemble=incumbent_ensemble,
        )

        # Evaluate candidates
        suggested_ensemble, suggested_pipeline = searcher.suggest(
            max_num_pipelines=max_num_pipelines,
            batch_size=batch_size,
            num_suggestion_batches=num_suggestion_batches,
            num_suggestions_per_batch=num_suggestions_per_batch,
            num_inner_epochs=num_inner_epochs,
        )

        _, observed_metric, _, _ = metadataset.evaluate_ensembles([suggested_ensemble])

        if max_num_pipelines > 1 and apply_posthoc_ensemble:
            post_hoc_ensemble, post_hoc_ensemble_metric = posthoc_ensembler.sample(
                X_obs=X_obs,
                max_num_pipelines=max_num_pipelines,
                num_suggestion_batches=num_suggestion_batches,
                num_suggestions_per_batch=num_suggestions_per_batch,
            )
            if post_hoc_ensemble_metric < observed_metric:
                suggested_ensemble = post_hoc_ensemble
                observed_metric = post_hoc_ensemble_metric

        # Update bookkeeping variables
        X_obs = np.concatenate((X_obs, [suggested_pipeline]))
        X_pending = np.setdiff1d(X_pending, [suggested_pipeline])

        if observed_metric < incumbent:
            incumbent = observed_metric.item()
            incumbent_ensemble = suggested_ensemble

        if wandb.run is not None:
            wandb.log({"searcher_iteration": iteration, "incumbent": incumbent})
            wandb.log(
                {
                    "searcher_iteration": iteration,
                    "incumbent (norm)": metadataset.compute_normalized_score(
                        torch.tensor(incumbent)
                    ),
                }
            )

        if X_pending.size == 0:
            break

    if wandb.run is not None:
        wandb.log(
            {
                "incumbent_ensemble": incumbent_ensemble,
                "incumbent_ensemble_metric": incumbent,
            }
        )
