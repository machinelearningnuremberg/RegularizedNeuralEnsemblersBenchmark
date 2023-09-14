import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import rankdata

import os

base_font_size = 14
until_iteration = 100
current_file_path = os.path.dirname(os.path.abspath(__file__))

api = wandb.Api()
user_name = "spinedaa"
project_name = "SearchingOptimalEnsembles"
group_name = "RS00"
output_folder = "saved_plots"
os.makedirs(os.path.join(current_file_path, output_folder), exist_ok=True)

# Dictionary to hold results
results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))





group_names = ["RS00", "DRE03", "DRE04", "DRE05", "DRE07"]

for group_name in group_names:
    print("w")
    # Fetch runs
    runs = api.runs(f"{user_name}/{project_name}",
                    filters={"$and": [{"group": group_name}, {"state": "finished"}]}
                    )

    for run in runs:
        history = run.history(keys=["incumbent (norm)", "searcher_iteration"], pandas=False)
        max_num_pipelines = run.config["max_num_pipelines"]
        dataset_id = run.config["dataset_id"]

        if history:
            incumbent_values = [record["incumbent (norm)"] for record in history]
            iteration_values = [record["searcher_iteration"] for record in history]

            results[max_num_pipelines][group_name][dataset_id] = incumbent_values
    
until_iteration = 100
num_datasets = 6
num_groups = len(group_names)
max_num_pipelines_values = [1,2,4,6,8,10]

for max_num_pipelines in max_num_pipelines_values:
    results_matrix = []

    for dataset_id in range(0, num_datasets):
        temp_results = []
        omit_dataset = False
        for group_name in group_names:
            if dataset_id in results[max_num_pipelines][group_name].keys():
                incumbent_values = results[max_num_pipelines][group_name][dataset_id]
            else:
                omit_dataset = True
            if len(incumbent_values) == until_iteration:
                temp_results.append(incumbent_values)
            else:
                print("Problem with dataset_id:", dataset_id, "in group_name:", group_name)
        if not omit_dataset:
            results_matrix.append(temp_results)
        else:
            print("Omitting dataset_id:", dataset_id)


    results_matrix = np.array(results_matrix)

    regret = results_matrix.mean(axis=0)

    if results_matrix.shape[0] == 0:
        print("Omitting max_num_pipelines:", max_num_pipelines)
        continue

    rank = rankdata(results_matrix, axis=1).mean(axis=0)

    plt.figure()
    plt.plot(rank.T)
    plt.legend(group_names)
    plt.savefig(os.path.join(current_file_path, output_folder, f"rank2_{max_num_pipelines}.png"))

    plt.figure()
    plt.plot(regret.T)
    plt.legend(group_names)
    plt.savefig(os.path.join(current_file_path, output_folder, f"regret2_{max_num_pipelines}.png"))


group_name = "DRE03"

results_matrix = []

max_num_pipelines_values = [1,2,4,6,8,10]
for dataset_id in range(0, num_datasets):
    temp_results = []
    omit_dataset = False
    for max_num_pipelines in max_num_pipelines_values:
        if dataset_id in results[max_num_pipelines][group_name].keys():
            incumbent_values = results[max_num_pipelines][group_name][dataset_id]
        else:
            omit_dataset = True
        if len(incumbent_values) == until_iteration:
            temp_results.append(incumbent_values)
        else:
            print("Problem with dataset_id:", dataset_id, "in group_name:", group_name)
    if not omit_dataset:
        results_matrix.append(temp_results)
    else:
        print("Omitting dataset_id:", dataset_id)


results_matrix = np.array(results_matrix)

regret = results_matrix.mean(axis=0)

rank = rankdata(results_matrix, axis=1).mean(axis=0)

plt.figure()
plt.plot(rank.T)
plt.legend(max_num_pipelines_values)
plt.savefig(os.path.join(current_file_path, output_folder, f"{group_name}_max_num_pipelines_rank.png"))

plt.figure()
plt.plot(regret.T)
plt.legend(max_num_pipelines_values)
plt.savefig(os.path.join(current_file_path, output_folder, f"{group_name}_max_num_pipelines_regret.png"))


group_name = "DRE02"
results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
runs = api.runs(f"{user_name}/{project_name}",
                filters={"$and": [{"group": group_name}, {"state": "finished"}]}
                )

for run in runs:
    history = run.history(keys=["incumbent (norm)", "searcher_iteration"], pandas=False)
    max_num_pipelines = run.config["max_num_pipelines"]
    dataset_id = run.config["dataset_id"]
    config_id = run.name

    if history:
        incumbent_values = [record["incumbent (norm)"] for record in history]
        iteration_values = [record["searcher_iteration"] for record in history]

        results[config_id][dataset_id] = incumbent_values


results_matrix = []
config_ids = list(results.keys())
config_ids.remove("dre_5")
config_ids.remove(5)
config_ids = [config_ids[i] for i in [7, 30, 19, 27, 26, 10]]

for dataset_id in [0,2,3,5]:
    temp_results = []
    omit_dataset = False
    for name in config_ids:
        if dataset_id in results[name].keys():
            incumbent_values = results[name][dataset_id]
        else:
            omit_dataset = True
            print("Problem in dataset_id:", dataset_id, "in config_id:", name)
        if len(incumbent_values) == until_iteration:
            temp_results.append(incumbent_values)
        else:
            print("Problem with dataset_id:", dataset_id, "in group_name:", group_name)
    if not omit_dataset:
        results_matrix.append(temp_results)
    else:
        print("Omitting dataset_id:", dataset_id)


results_matrix = np.array(results_matrix)

regret = results_matrix.mean(axis=0)

rank = rankdata(results_matrix, axis=1).mean(axis=0)

plt.figure()
plt.plot(rank.T)
plt.legend(config_ids)
plt.savefig(os.path.join(current_file_path, output_folder, f"{group_name}_comparison.png"))

plt.figure()
plt.plot(regret.T)
plt.legend(config_ids)
plt.savefig(os.path.join(current_file_path, output_folder, f"{group_name}_comparison.png"))
