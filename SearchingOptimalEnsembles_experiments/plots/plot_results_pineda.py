import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.stats import rankdata
import itertools
import pandas as pd
from pathlib import Path

base_font_size = 14
until_iteration = 100
current_file_path = Path(os.path.dirname(os.path.abspath(__file__)))

api = wandb.Api()
user_name = "spinedaa"
project_name = "SOE_tabrepo"
group_name = "RS00"
output_folder = "saved_plots"
os.makedirs(os.path.join(current_file_path, output_folder), exist_ok=True)

# Dictionary to hold results
results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))


#group_names = ["RS00", "DRE03", "DRE04", "DRE05", "DRE07"]
group_names = ["neural12_0", "greedy7_0", "neural19_0"]
#group_names = ["neural16_0", "greedy9_2", "greedy9_1", "greedy9_0"]

group_dict = {
    "neural26_0": "Neural_32hd_None",
    "neural26_1": "Neural_16hd_None",
    "neural26_2": "Neural_32hd_random",
    "neural26_3": "Neural_16hd_random",
    "neural26_4": "Neural_32hd_leo",
    "neural26_5": "Neural_16hd_leo",
    "neural26_6": "Neural_32hd_divbo",
    "neural26_7": "Neural_16hd_divbo",
    "neural27_0": "Neural_32hd_None_simple",
    "neural27_3": "Neural_32hd_random_simple",
    "neural27_5": "Neural_32hd_leo_simple",
    "neural27_7": "Neural_32hd_divbo_simple",
    "greedy15_0": "Greedy_20hd_None",
    "greedy15_1": "Greedy_10hd_None",
    "greedy15_2": "Greedy_20hd_random",
    "greedy15_3": "Greedy_10hd_random",
    "greedy15_4": "Greedy_20hd_leo",
    "greedy15_5": "Greedy_10hd_leo",
    "greedy15_6": "Greedy_20hd_divbo",
    "greedy15_7": "Greedy_10hd_divbo",
    "leo4_0": "LEO",
    "random4_0": "Random",
    "divbo4_0": "DivBO"
}


# group_dict = {
#     "neural23_2": "Neural_32hd_random",
#     "greedy14_2": "Greedy_20hd_random",
#     "greedy14_3": "Greedy_10hd_random",
#     "leo3_0": "LEO",
#     "random3_0": "Random",
#     "divbo3_0": "DivBO"
# }

group_dict = {
    "neural26_2": "Neural_32hd_random",
    "greedy15_2": "Greedy_20hd_random",
    "leo4_0": "LEO",
    "random4_0": "Random",
    "divbo4_0": "DivBO"
}

download = False
test_metrics = []
for group_name, results_group_name in group_dict.items():
    print("Processing group name:", group_name)
    file_name = results_group_name + ".csv"
    if download: 
        temp_test_metrics = []
        # Fetch runs
        runs = api.runs(
            f"{user_name}/{project_name}",
            filters={"$and": [{"group": group_name}, {"state": "finished"}]},
        )
        
        for run in runs:
            history = run.history(
                keys=["incumbent_ensemble_test_metric", "incumbent_ensemble_metric"], pandas=False
            )
            max_num_pipelines = run.config["max_num_pipelines"]
            seed = str(run.config["seed"])
            dataset_id = run.config["dataset_id"]
            meta_split_id = run.config["meta_split_id"]
            history[0]["group_name"] = group_name
            history[0]["dataset_complete_id"] = f"{dataset_id}-{meta_split_id}"
            history[0].pop("_step")
            temp_test_metrics.append(history[0])
        
        test_metrics.extend(temp_test_metrics)
        pd.DataFrame(temp_test_metrics).to_csv(current_file_path / "saved_results" / file_name)
    else:
        data = pd.read_csv(current_file_path / "saved_results" / file_name)
        test_metrics.append(data)

if download:
    data = pd.DataFrame(test_metrics)
else:
    data = pd.concat(test_metrics)
data = pd.pivot_table(data, values="incumbent_ensemble_test_metric", index="dataset_complete_id", columns="group_name")
print(data.rank(axis=1).mean())
print(data.mean(axis=0))

