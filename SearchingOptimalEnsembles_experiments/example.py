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
    dataset_id: int,
    **run_args,  # pylint: disable=unused-argument
) -> None:
    config = {
        "worker_dir": worker_dir,
        "metadataset_name": metadataset_name,
        "searcher_name": searcher_name,
        "surrogate_name": surrogate_name,
        "acquisition_name": acquisition_name,
        "num_iterations": run_args["num_iterations"],
        "num_inner_epochs": run_args["num_inner_epochs"],
        "num_suggestion_batches": run_args["num_suggestion_batches"],
        "num_suggestions_per_batch": run_args["num_suggestions_per_batch"],
        "meta_num_epochs": run_args["meta_num_epochs"],
        "max_num_pipelines": run_args["max_num_pipelines"],
        "dataset_id": dataset_id,
    }

    if surrogate_args is not None:
        config.update(surrogate_args)

    if acquisition_args is not None:
        config.update(acquisition_args)

    if wandb.run is not None:
        wandb.config.update(config)

    SOE.run(
        worker_dir=worker_dir,
        metadataset_name=metadataset_name,
        searcher_name=searcher_name,
        surrogate_name=surrogate_name,
        surrogate_args=surrogate_args,
        acquisition_name=acquisition_name,
        acquisition_args=acquisition_args,
        dataset_id=dataset_id,
        run_args=run_args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_dir",
        type=str,
        default="/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments",
    )
    ##############################################################################
    parser.add_argument("--metadataset_name", type=str, default="quicktune")
    parser.add_argument("--searcher_name", type=str, default="bo")
    ##############################################################################
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--max_num_pipelines", type=int, default=5)
    ##############################################################################
    parser.add_argument("--meta_num_epochs", type=int, default=0)
    ##############################################################################
    parser.add_argument("--num_inner_epochs", type=int, default=100)
    parser.add_argument("--num_suggestion_batches", type=int, default=5)
    parser.add_argument("--num_suggestions_per_batch", type=int, default=1000)
    ##############################################################################
    parser.add_argument("--dataset_id", type=int, default=0)
    ##############################################################################
    parser.add_argument("--experiment_group", type=str, default="ablate_DKL")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="debug")
    ############################ COMMON SURROGATE ARGS ############################
    parser.add_argument("--surrogate_name", type=str, default="dkl")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    ############################## DRE SURROGATE ARGS #############################
    parser.add_argument("--criterion_type", type=str, default="weighted_listwise")
    parser.add_argument("--activation_output", type=str, default="sigmoid")
    parser.add_argument("--score_with_rank", type=bool, default=False)
    ############################## DKL SURROGATE ARGS #############################
    parser.add_argument("--kernel_name", type=str, default="matern")
    parser.add_argument("--ard", type=bool, default=False)
    parser.add_argument("--nu", type=float, default=2.5)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--optional_dim", type=int, default=None)
    ############################### ACQUISITION ARGS ##############################
    parser.add_argument("--acquisition_name", type=str, default="ei")
    parser.add_argument("--beta", type=float, default=0.0)
    ##############################################################################
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=args.log_level.upper())

    try:
        wandb.init(
            project="SearchingOptimalEnsembles",
            group=args.experiment_group,
            name=args.run_name,
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

    if args.surrogate_name == "dkl":
        surrogate_args = {
            "kernel_name": args.kernel_name,
            "ard": args.ard,
            "nu": args.nu,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_seeds": args.num_seeds,
            "lr": args.lr,
            # "optional_dim": args.optional_dim,
        }
    elif args.surrogate_name == "dre":
        surrogate_args = {
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_seeds": args.num_seeds,
            "criterion_type": args.criterion_type,
            "activation_output": args.activation_output,
            "score_with_rank": args.score_with_rank,
        }
    else:
        raise ValueError(f"Unknown surrogate name: {args.surrogate_name}")

    acquisition_args = {"beta": args.beta}

    run_args = {
        "meta_num_epochs": args.meta_num_epochs,
        "num_iterations": args.num_iterations,
        "num_inner_epochs": args.num_inner_epochs,
        "num_suggestion_batches": args.num_suggestion_batches,
        "num_suggestions_per_batch": args.num_suggestions_per_batch,
        "max_num_pipelines": args.max_num_pipelines,
    }

    run(
        worker_dir=args.worker_dir,
        metadataset_name=args.metadataset_name,
        searcher_name=args.searcher_name,
        surrogate_name=args.surrogate_name,
        surrogate_args=surrogate_args,
        acquisition_name=args.acquisition_name,
        acquisition_args=acquisition_args,
        dataset_id=args.dataset_id,
        **run_args,
    )
