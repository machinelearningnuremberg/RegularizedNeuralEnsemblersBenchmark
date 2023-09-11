from __future__ import annotations

import argparse
import logging

import wandb

import SearchingOptimalEnsembles as SOE

from SearchingOptimalEnsembles_experiments.utils.util import set_seed


def run(
    worker_dir: str,
    metadataset_name: str,
    searcher_name: str,
    surrogate_name: str,
    acquisition_name: str,
    surrogate_args: dict | None,
    acquisition_args: dict | None,
    experiment_id: str | None = None,
    group_id: str | None = None,
    dataset_id: int = 0,
    log_wandb: bool = True,
    **run_args,  # pylint: disable=unused-argument
) -> None:

    config = {"worker_dir": worker_dir,
            "metadataset_name": metadataset_name,
            "searcher_name": searcher_name,
            "surrogate_name": surrogate_name,
            "experiment_id": experiment_id,
            "group_id": group_id,
            "num_iterations": run_args["num_iterations"],
            "num_inner_epochs": run_args["num_inner_epochs"],
            "num_suggestion_batches": run_args["num_suggestion_batches"],
            "num_suggestions_per_batch": run_args["num_suggestions_per_batch"],
            "meta_num_epochs": run_args["meta_num_epochs"],
            "max_num_pipelines": run_args["max_num_pipelines"],
            "dataset_id": dataset_id
            }

    if surrogate_args is not None:
        config.update(surrogate_args)

    if acquisition_args is not None:
        config.update(acquisition_args)

    if log_wandb:
        try:
            wandb.init(
                name=experiment_id,
                project="SearchingOptimalEnsembles",
                group=group_id,
                config=config
            )
        except wandb.errors.UsageError:
            print("Wandb is not available")

    SOE.run(
        worker_dir=worker_dir,
        metadataset_name=metadataset_name,
        searcher_name=searcher_name,
        surrogate_name=surrogate_name,
        acquisition_name=acquisition_name,
        surrogate_args=surrogate_args,
        acquisition_args=acquisition_args,
        dataset_id=dataset_id,
        run_args=run_args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_dir",
        type=str,
         default="/home/pineda/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments",
        #default="/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments",
    )
    parser.add_argument("--metadataset_name", type=str, default="quicktune")
    parser.add_argument("--searcher_name", type=str, default="bo")
    parser.add_argument("--surrogate_name", type=str, default="dre")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_num_pipelines", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--num_inner_epochs", type=int, default=100)
    parser.add_argument("--log_level", type=str, default="debug")
    parser.add_argument("--criterion_type", type=str, default="weighted_listwise")
    parser.add_argument("--num_suggestion_batches", type=int, default=5)
    parser.add_argument("--num_suggestions_per_batch", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--activation_output", type=str, default="sigmoid")
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--score_with_rank", type=int, default=0)
    parser.add_argument("--group_id", type=str, default="exp0")
    parser.add_argument("--acquisition_name", type=str, default="ei")
    parser.add_argument("--dataset_id", type=int, default=0)
    parser.add_argument("--log-wandb", action="store_true")


    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    surrogate_args = None
    if args.surrogate_name == "dre":
        surrogate_args = {
            "criterion_type": args.criterion_type,
            "activation_output": args.activation_output,
            "score_with_rank": bool(args.score_with_rank),
        }

    acquisition_args = {"beta": args.beta}

    run_args = {
        "num_iterations": args.num_iterations,
        "max_num_pipelines": args.max_num_pipelines,
        "meta_num_epochs": 0,
        "num_inner_epochs": args.num_inner_epochs,
        "num_suggestion_batches": args.num_suggestion_batches,
        "num_suggestions_per_batch": args.num_suggestions_per_batch,
    }

    run(
        worker_dir=args.worker_dir,
        metadataset_name=args.metadataset_name,
        searcher_name=args.searcher_name,
        surrogate_name=args.surrogate_name,
        acquisition_name=args.acquisition_name,
        surrogate_args=surrogate_args,
        acquisition_args=acquisition_args,
        experiment_id=args.experiment_id,
        group_id=args.group_id,
        dataset_id=args.dataset_id,
        log_wandb=args.log_wandb,
        **run_args,
    )
