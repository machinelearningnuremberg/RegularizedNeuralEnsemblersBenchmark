from __future__ import annotations

import argparse
import logging

import wandb

import SearchingOptimalEnsembles as SOE

from .utils.util import set_seed


def run(metadataset_name: str, surrogate_name: str) -> None:
    try:
        wandb.init(
            project="SearchingOptimalEnsembles",
            group=f"{metadataset_name}_{surrogate_name}",
        )
    except wandb.errors.UsageError:
        print("Wandb is not available")

    SOE.run(metadataset_name=metadataset_name, surrogate_name=surrogate_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadataset_name", type=str, default="quicktune")
    parser.add_argument("--surrogate_name", type=str, default="dkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="debug")
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=args.log_level.upper())

    run(metadataset_name=args.metadataset_name, surrogate_name=args.surrogate_name)
