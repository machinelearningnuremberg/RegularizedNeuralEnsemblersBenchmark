from __future__ import annotations

import inspect
import random

import numpy as np
import torch


def set_seed(seed: int):
    """
    Set the seeds for all used libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


COMPLEX_KEYS = {
    "surrogate_args": {
        "dkl": [
            "kernel_name",
            "ard",
            "nu",
            "add_y",
            "hidden_dim",
            "out_dim",
            "num_heads",
            "num_seeds",
            "lr",
        ],
        "dre": [
            "add_y",
            "hidden_dim",
            "out_dim",
            "num_heads",
            "num_seeds",
            "num_layers_ff",
            "criterion_type",
            "activation_output",
            "score_with_rank",
            "lr",
        ],
        "rf": ["n_estimators"],
        "gp": [],
        "lightgbm": [],
    },
    "acquisition_args": {
        "ei": ["beta"],
        "lcb": ["beta"],
    },
}


def extract_complex_keys(args, main_key, specific_key):
    """Extract specific complex keys from args based on COMPLEX_KEYS definition."""
    specific_sub_keys = COMPLEX_KEYS[main_key].get(specific_key, [])
    return {key: getattr(args, key) for key in specific_sub_keys}


def get_config(args, function) -> dict:
    # First, make a shallow copy of args dictionary
    all_args = vars(args).copy()

    # Get the argument names for the run() function
    run_signature = inspect.signature(function)
    run_keys = set(run_signature.parameters.keys())

    # Separate basic and complex keys
    basic_config = {key: value for key, value in all_args.items() if key in run_keys}

    # Add the specific complex configurations
    basic_config["surrogate_args"] = extract_complex_keys(
        args, "surrogate_args", args.surrogate_name
    )
    basic_config["acquisition_args"] = extract_complex_keys(
        args, "acquisition_args", args.acquisition_name
    )

    return basic_config
