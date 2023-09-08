import os
from pathlib import Path

ARGS_PATH = Path(__file__).parent
EXP_PATH = ARGS_PATH.parent.parent
OUTPUT_DIR = "/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/"

SURROGATE_NAMES = ["dkl"]
METADATA_NAMES = ["quicktune"]
SEEDS = [42, 43, 44, 45, 46, 47, 48]
RUN_IDS = [0, 1, 2, 3, 4, 5]
# SURROGATE_NAMES = ["dkl", "dre"]
# METADATA_NAMES = ["quicktune", "scikit-learn", "nasbench201"]

with open(os.path.join(ARGS_PATH, "all_experiments.txt"), "w+", encoding="UTF-8") as f:
    for metadataset_name in METADATA_NAMES:
        for surrogate_name in SURROGATE_NAMES:
            for run_id in RUN_IDS:
                for seed in SEEDS:
                    BASE_CMD = f"python {EXP_PATH}/example.py"
                    cmd = BASE_CMD
                    cmd += f" --surrogate_name {surrogate_name}"
                    cmd += f" --metadataset_name {metadataset_name}"
                    cmd += f" --seed {seed}"
                    cmd += f" --run_id {run_id}"

                    f.writelines(cmd + "\n")
