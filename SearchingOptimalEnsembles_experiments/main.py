from __future__ import annotations

import argparse
import logging

import wandb

import SearchingOptimalEnsembles as SOE
from SearchingOptimalEnsembles_experiments.utils.util import get_config, set_seed

TAG_KEYS = [
    "seed",
    "metadataset_name",
    "searcher_name",
    "surrogate_name",
    "acquisition_name",
    "meta_num_epochs",
    "max_num_pipelines",
]

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
    parser.add_argument("--initial_design_size", type=int, default=5)
    ##############################################################################
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--max_num_pipelines", type=int, default=5)
    ##############################################################################
    parser.add_argument("--meta_num_epochs", type=int, default=0)
    parser.add_argument("--meta_num_inner_epochs", type=int, default=1)
    ##############################################################################
    parser.add_argument("--num_inner_epochs", type=int, default=100)
    parser.add_argument("--num_suggestion_batches", type=int, default=5)
    parser.add_argument("--num_suggestions_per_batch", type=int, default=1000)
    ##############################################################################
    parser.add_argument("--experiment_group", type=str, default="TEST")
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
    parser.add_argument("--sampler_name", type=str, default="random")
    parser.add_argument("--ensembler_name", type=str, default="random")
    ##############################################################################
    parser.add_argument("--dataset_id", type=int, default=0)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=args.log_level.upper())
    print(args)

    if not args.no_wandb:
        try:
            wandb.init(
                project="SearchingOptimalEnsembles",
                group=args.experiment_group,
                name=args.run_name,
            )
            for tag_key in TAG_KEYS:
                wandb.run.tags += (f"{tag_key}={vars(args)[tag_key]}",)
        except wandb.errors.UsageError:
            print("Wandb is not available")

    args.worker_dir = f"{args.worker_dir}/{args.experiment_group}"

    config = get_config(args, function=SOE.run)

    # Update wandb config
    if wandb.run is not None:
        wandb.config.update(config)

    SOE.run(**config)
