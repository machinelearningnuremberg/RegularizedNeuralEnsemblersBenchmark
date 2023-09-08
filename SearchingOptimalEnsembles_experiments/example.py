from __future__ import annotations

import argparse
import logging

import wandb

import SearchingOptimalEnsembles as SOE

from .utils.util import set_seed


def run(
    worker_dir: str,
    metadataset_name: str,
    searcher_name: str,
    surrogate_name: str,
    surrogate_args: dict | None,
    acquisition_name: str,
    acquisition_args: dict | None,
    meta_num_epochs: int,
    max_num_pipelines: int,
    dataset_id: int,
) -> None:
    SOE.run(
        worker_dir=worker_dir,
        metadataset_name=metadataset_name,
        searcher_name=searcher_name,
        surrogate_name=surrogate_name,
        surrogate_args=surrogate_args,
        acquisition_name=acquisition_name,
        acquisition_args=acquisition_args,
        dataset_id=dataset_id,
        meta_num_epochs=meta_num_epochs,
        max_num_pipelines=max_num_pipelines,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_dir",
        type=str,
        default="/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments",
    )
    parser.add_argument("--metadataset_name", type=str, default="quicktune")
    parser.add_argument("--searcher_name", type=str, default="bo")
    parser.add_argument("--surrogate_name", type=str, default="dkl")
    parser.add_argument("--acquisition_name", type=str, default="ei")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_id", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="debug")
    parser.add_argument("--experiment_group", type=str, default="debug")
    parser.add_argument("--meta_num_epochs", type=int, default=0)
    parser.add_argument("--max_num_pipelines", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=args.log_level.upper())

    try:
        wandb.init(
            project="SearchingOptimalEnsembles",
            group=args.experiment_group,
        )
        wandb.run.tags += (f"seed={args.seed}",)
        wandb.run.tags += (f"metadataset_name={args.metadataset_name}",)
        wandb.run.tags += (f"searcher_name={args.searcher_name}",)
        wandb.run.tags += (f"surrogate_name={args.surrogate_name}",)
        wandb.run.tags += (f"acquisition_name={args.acquisition_name}",)
        wandb.run.tags += (f"meta_num_epochs={args.meta_num_epochs}",)
        wandb.run.tags += (f"max_num_pipelines={args.max_num_pipelines}",)
    except wandb.errors.UsageError:
        print("Wandb is not available")

    args.worker_dir = f"{args.worker_dir}/{args.experiment_group}"

    surrogate_args = None
    acquisition_args = None

    run(
        worker_dir=args.worker_dir,
        metadataset_name=args.metadataset_name,
        searcher_name=args.searcher_name,
        surrogate_name=args.surrogate_name,
        surrogate_args=surrogate_args,
        acquisition_name=args.acquisition_name,
        acquisition_args=acquisition_args,
        dataset_id=args.dataset_id,
        meta_num_epochs=args.meta_num_epochs,
        max_num_pipelines=args.max_num_pipelines,
    )
