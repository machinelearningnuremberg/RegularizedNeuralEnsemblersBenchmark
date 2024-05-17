from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from typing_extensions import Literal

import wandb

from .metadatasets import MetaDatasetMapping
from .metadatasets.base_metadataset import META_SPLITS
from .posthoc import EnsemblerMapping
from .searchers import SearcherMapping
from .utils.common import instance_from_map
from .utils.eval import eval
from .utils.logger import get_logger


def run(
    worker_dir: str,
    metadataset_name: Literal["scikit-learn", "nasbench201", "quicktune"],
    #############################################
    searcher_name: Literal["random", "bo", "None"] = "bo",
    initial_design_size: int = 5,
    #############################################
    surrogate_name: Literal["dkl", "dre", "rf", "gp"] = "dkl",
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
    apply_posthoc_ensemble_each_iter: bool = False,
    apply_posthoc_ensemble_at_end: bool = True,
    #############################################
    ne_hidden_dim: int = 512,
    ne_context_size: int = 32,
    ne_reg_term_div: float = 0.1,
    ne_add_y: bool = True,
    #############################################
    dataset_id: int = 0,
    meta_split_id: int = 0,
    data_version: str = "micro",
    metric_name: str = "nll",
    device: str = "cuda",
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

    logger = get_logger(name="SEO-MAIN", logging_level="debug")

    metadataset_args = {
        "meta_split_ids": META_SPLITS[meta_split_id],
        "metric_name": metric_name,
        "data_version": data_version,
    }

    metadataset = instance_from_map(
        MetaDatasetMapping,
        metadataset_name,
        name="metadataset",
        kwargs=metadataset_args,
    )

    if checkpoint_path is None:
        checkpoint_dir = Path(worker_dir) / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_path)
    # Create the checkpoint directory if it does not exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if searcher_name != "None":
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
    else:
        searcher = None

    ensembler_args = {
        "metadataset": metadataset,
        "device": device,
        "ne_hidden_dim": ne_hidden_dim,
        "ne_context_size": ne_context_size,
        "ne_reg_term_div": ne_reg_term_div,
        "ne_add_y": ne_add_y,
    }
    posthoc_ensembler = instance_from_map(
        EnsemblerMapping,
        ensembler_name,
        name="posthoc_ensembler",
        kwargs=ensembler_args,
    )

    if searcher is not None:
        if meta_num_epochs > 0:
            assert hasattr(
                searcher, "surrogate"
            ), "Surrogate must be defined for meta-training."
            # TODO: assert that the surrogate can be meta_trained

            logger.debug("Meta-training the surrogate...")

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

        logger.debug(f"Using dataset: {dataset_name}")

        # if wandb.run is not None:
        #     wandb.config.update({"dataset_id": dataset_name}, allow_val_change=True)
        searcher.sampler.set_state(dataset_name=dataset_name, meta_split="meta-test")

        logger.debug("Sampling initial design...")

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

        if wandb.run is not None:
            wandb.log({"searcher_iteration": 0, "incumbent": incumbent})
            wandb.log(
                {
                    "searcher_iteration": 0,
                    "incumbent (norm)": metadataset.compute_normalized_score(
                        torch.tensor(incumbent)
                    ),
                }
            )

        logger.info(f"Initial incumbent: {incumbent}")
        logger.debug("Starting search...")

        # Main search loop
        for iteration in range(num_iterations):
            searcher.set_state(
                X_obs=X_obs,
                X_pending=X_pending,
                incumbent=incumbent,
                incumbent_ensemble=incumbent_ensemble,
                iteration=iteration,
            )

            # Evaluate candidates
            suggested_ensemble, suggested_pipeline = searcher.suggest(
                max_num_pipelines=max_num_pipelines,
                batch_size=batch_size,
                num_suggestion_batches=num_suggestion_batches,
                num_suggestions_per_batch=num_suggestions_per_batch,
                num_inner_epochs=num_inner_epochs,
            )

            _, observed_metric, _, _ = metadataset.evaluate_ensembles(
                [suggested_ensemble]
            )

            if max_num_pipelines > 1 and apply_posthoc_ensemble_each_iter:
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

                logger.info(
                    f"Iteration: {iteration}/{num_iterations} - New incumbent: {incumbent}"
                )
            else:
                logger.info(f"Iteration: {iteration}/{num_iterations}")

            if wandb.run is not None:
                wandb.log({"searcher_iteration": iteration + 1, "incumbent": incumbent})
                wandb.log(
                    {
                        "searcher_iteration": iteration + 1,
                        "incumbent (norm)": metadataset.compute_normalized_score(
                            torch.tensor(incumbent)
                        ),
                    }
                )

            if X_pending.size == 0:
                break

    else:
        dataset_name = metadataset.meta_splits["meta-test"][dataset_id]
        metadataset.set_state(dataset_name=dataset_name)
        num_existing_pipelines = len(metadataset.hp_candidates_ids)
        X_obs = np.arange(num_existing_pipelines)
        np.random.shuffle(X_obs)

    if apply_posthoc_ensemble_at_end:
        incumbent_ensemble, incumbent = posthoc_ensembler.sample(
            X_obs, max_num_pipelines=max_num_pipelines
        )

    test_metric, test_metric_per_pipeline = eval(
        incumbent_ensemble,
        metadataset_name=metadataset_name,
        dataset_id=dataset_id,
        meta_split_id=meta_split_id,
        data_version=data_version,
        metric_name=metric_name,
        split="test",
        ensembler=posthoc_ensembler,
    )

    if wandb.run is not None:
        wandb.log(
            {
                "incumbent_ensemble": incumbent_ensemble,
                "incumbent_ensemble_metric": incumbent,
                "incumbent_ensemble_test_metric": test_metric,
            }
        )
    else:
        print(
            {
                "incumbent_ensemble": incumbent_ensemble,
                "incumbent_ensemble_metric": incumbent,
                "incumbent_ensemble_test_metric": test_metric,
            }
        )
