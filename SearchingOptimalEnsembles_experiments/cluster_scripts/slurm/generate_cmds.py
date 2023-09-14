from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

ARGS_PATH = Path(__file__).parent
EXP_PATH = ARGS_PATH.parent.parent
OUTPUT_DIR = "/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/"

# SURROGATE_NAMES = ["dkl"]
# METADATA_NAMES = ["quicktune"]
# SEEDS = [42, 43, 44, 45, 46, 47, 48]
# RUN_IDS = [0, 1, 2, 3, 4, 5]
# # SURROGATE_NAMES = ["dkl", "dre"]
# # METADATA_NAMES = ["quicktune", "scikit-learn", "nasbench201"]

# with open(os.path.join(ARGS_PATH, "all_experiments.txt"), "w+", encoding="UTF-8") as f:
#     for metadataset_name in METADATA_NAMES:
#         for surrogate_name in SURROGATE_NAMES:
#             for run_id in RUN_IDS:
#                 for seed in SEEDS:
#                     BASE_CMD = f"python {EXP_PATH}/example.py"
#                     cmd = BASE_CMD
#                     cmd += f" --surrogate_name {surrogate_name}"
#                     cmd += f" --metadataset_name {metadataset_name}"
#                     cmd += f" --seed {seed}"
#                     cmd += f" --run_id {run_id}"

#                     f.writelines(cmd + "\n")

EXPERIMENT_GROUP = "ablate_DKL_1"

# config_space: Dict[str, List[Any]] = {
#     "kernel_name": ["matern", "rbf"],
#     "ard": [False, True],
#     "nu": [2.5, 1.5],
#     "hidden_dim": [32, 64, 128, 256],
#     "num_heads": [2, 4, 8],
#     "num_seeds": [1, 2, 4],
#     "lr": [1e-3, 1e-4, 1e-5],
#     "acquisition_name": ["ei", "lcb"],
#     "beta": [0., 0.01, 0.1, 1],
#     # "optional_dim": [None]
# }

config_space: dict[str, list[Any]] = {
    "kernel_name": ["matern"],
    "ard": [True],
    "nu": [2.5, 1.5],
    "hidden_dim": [32, 64, 128, 256],
    "num_heads": [2, 4, 8, 16],
    "num_seeds": [1],
    "lr": [1e-3, 1e-4, 1e-5],
    "acquisition_name": ["ei"],
    "beta": [1],
    # "optional_dim": [None]
}

with open(
    os.path.join(ARGS_PATH, f"{EXPERIMENT_GROUP}.txt"), "w+", encoding="UTF-8"
) as f:
    for i in range(20):
        selected_hps: dict = {
            key: random.choice(value) for key, value in config_space.items()
        }

        run_name = "DKL_"
        for key, value in selected_hps.items():
            run_name += f"{value}_"

        cmd = f"python {EXP_PATH}/main.py --max_num_pipelines 5"
        cmd += " --surrogate_name dkl"

        for key, value in selected_hps.items():
            cmd += f" --{key} {value}"

        cmd += f" --experiment_group {EXPERIMENT_GROUP}"

        for j in range(6):
            temp_command = cmd
            temp_command += f" --dataset_id {j}"
            temp_command += f" --run_name {run_name}_{j}"
            f.writelines(temp_command + "\n")
