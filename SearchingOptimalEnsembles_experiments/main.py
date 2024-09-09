from __future__ import annotations

import argparse
import logging
import warnings
import os
import json

os.environ["WANDB_DIR"] = "/work/dlclarge1/pineda-seo_data/logs"

import SearchingOptimalEnsembles as SOE
import wandb
from SearchingOptimalEnsembles_experiments.utils.util import get_config, set_seed

warnings.filterwarnings("ignore", category=UserWarning, message="[LightGBM]")

TAG_KEYS = [
    "seed",
    "metadataset_name",
    "searcher_name",
    "surrogate_name",
    "acquisition_name",
    "meta_num_epochs",
    "max_num_pipelines",
    "sampler_name",
    "ensembler_name",
    "dataset_id",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_dir",
        type=str,
        #default="/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/",
        default="/work/dlclarge2/janowski-quicktune/results/"
    )
    ##############################################################################
    parser.add_argument("--metadataset_name", type=str, default="quicktune")
    parser.add_argument("--searcher_name", type=str, default="bo")
    parser.add_argument("--initial_design_size", type=int, default=1)
    ##############################################################################
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--max_num_pipelines_for_searcher", type=int, default=10)
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
    parser.add_argument("--surrogate_name", type=str, default="rf")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    ############################## DRE SURROGATE ARGS #############################
    parser.add_argument("--criterion_type", type=str, default="weighted_listwise")
    parser.add_argument("--activation_output", type=str, default="sigmoid")
    parser.add_argument("--score_with_rank", type=bool, default=False)
    parser.add_argument("--num_layers_ff", type=int, default=1)
    ############################## DKL SURROGATE ARGS #############################
    parser.add_argument("--kernel_name", type=str, default="matern")
    parser.add_argument("--ard", type=bool, default=True)
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--num_base_pipelines", type=int, default=20)
    parser.add_argument("--optional_dim", type=int, default=None)
    ############################## LEO ARGS #######################################
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--leo_surrogate_name", type=str, default="RF")
    ############################### ACQUISITION ARGS ##############################
    parser.add_argument("--acquisition_name", type=str, default="ei")
    parser.add_argument("--beta", type=float, default=0.0)
    ###############################NEURAL ENSEMBLER ARGS #########################
    parser.add_argument("--ne_learning_rate", type=float, default=0.0001)
    parser.add_argument("--ne_hidden_dim", type=int, default=32)
    parser.add_argument("--ne_reg_term_norm", type=float, default=0.)
    parser.add_argument("--ne_context_size", type=int, default=32)
    parser.add_argument("--ne_eval_context_size", type=int, default=256)
    parser.add_argument("--ne_reg_term_div", type=float, default=0.)
    parser.add_argument("--ne_add_y", action="store_true")
    parser.add_argument("--ne_use_context", action="store_true")
    parser.add_argument("--ne_mode", type=str, default="inference")
    parser.add_argument("--ne_num_layers", type=int, default=3)
    parser.add_argument("--ne_dropout_rate", type=float, default=0.)
    parser.add_argument("--ne_net_type", type=str, default="ffn")
    parser.add_argument("--ne_auto_dropout", action="store_true")
    parser.add_argument("--ne_weight_thd", type=float, default=0.)
    parser.add_argument("--ne_dropout_dist", type=str, default=None)
    parser.add_argument("--ne_omit_output_mask", action="store_true")
    parser.add_argument("--ne_batch_size", type=int, default=2048)
    parser.add_argument("--ne_net_mode", type=str, default="model_averaging")
    parser.add_argument("--ne_epochs", type=int, default=1000)
    parser.add_argument("--ne_patience", type=int, default=-1)
    ##################### OTHERS #####################################################
    parser.add_argument("--des_method_name", type=str, default="KNOP")
    parser.add_argument("--sks_model_name", type=str, default="random_forest")
    ##############################################################################
    parser.add_argument("--sampler_name", type=str, default="random")
    parser.add_argument("--ensembler_name", type=str, default="random")
    ##############################################################################
    parser.add_argument("--dataset_id", type=int, default=0)
    parser.add_argument("--meta_split_id", type=int, default=0)
    parser.add_argument("--metric_name", type=str, default="nll")
    parser.add_argument("--data_version", type=str, default="micro")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--normalize_performance", action="store_true")
    parser.add_argument("--apply_posthoc_ensemble_each_iter", action="store_true")
    parser.add_argument("--apply_posthoc_ensemble_at_end", action="store_true")
    parser.add_argument("--project_name", type=str, default="SearchingOptimalEnsembles")
    parser.add_argument("--job_id", type=str, default="default")

    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=args.log_level.upper())
    print(args)
    if not args.no_wandb:
        try:
            wandb.init(
                project=args.project_name,
                group=args.experiment_group,
                name=args.run_name,
            )
            for tag_key in TAG_KEYS:
                wandb.run.tags += (f"{tag_key}={vars(args)[tag_key]}",)
        except wandb.errors.UsageError:
            print("Wandb is not available")

    args.worker_dir = f"{args.worker_dir}/{args.project_name}/{args.experiment_group}/{args.dataset_id}/{args.seed}"

    config = get_config(args, function=SOE.run)

    # Update wandb config
    if wandb.run is not None:
        wandb.config.update(config)
        wandb.config.update({"seed": args.seed})

    results = SOE.run(**config)

    with open(args.worker_dir + '/results.json', 'w') as f:
        json.dump(results, f)
    
    with open(args.worker_dir + '/args.json', 'w') as f:
        json.dump(args.__dict__, f)


