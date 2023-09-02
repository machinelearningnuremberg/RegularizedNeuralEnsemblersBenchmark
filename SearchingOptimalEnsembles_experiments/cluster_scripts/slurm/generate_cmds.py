import os
from pathlib import Path

ARGS_PATH = Path(__file__).parent
EXP_PATH = ARGS_PATH.parent.parent
OUTPUT_DIR = "/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/"

SURROGATE_NAMES = ["dkl", "dre"]
METADATA_NAMES = ["scikit-learn"]
# SURROGATE_NAMES = ["dkl", "dre"]
# METADATA_NAMES = ["quicktune", "scikit-learn", "nasbench201"]

with open(os.path.join(ARGS_PATH, "all_experiments.txt"), "w+", encoding="UTF-8") as f:
    for metadataset_name in METADATA_NAMES:
        for surrogate_name in SURROGATE_NAMES:
            BASE_CMD = f"python {EXP_PATH}/example.py"
            cmd = BASE_CMD
            cmd += f" --surrogate_name {surrogate_name}"
            cmd += f" --metadataset_name {metadataset_name}"

            f.writelines(cmd + "\n")
