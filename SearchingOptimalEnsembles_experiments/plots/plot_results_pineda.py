import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.stats import rankdata
import itertools
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


#group_names = ["RS00", "DRE03", "DRE04", "DRE05", "DRE07"]
group_names = ["RS01_1", "LEO02_1", "DIVBO01_1", "DRE11_1", "DKL03_1"]
group_names = ["RS01_0", "LEO02_0", "DIVBO01_0", "DRE11_0", "DKL03_0"]
group_names = ["DRE15_1", "DRE15_5","DRE15_9", "DRE15_13", "DRE15_19", "DRE15_23", "DRE16_23", "DRE16_5"]
#group_names = ["DRE16_1", "DRE16_5","DRE16_9", "DRE16_13", "DRE16_19", "DRE16_23"]
group_names = ["DRE15_5", "DRE15_19", "DRE15_23", "DRE16_23", "DRE16_5"]

group_names = [f"DRE16_{i}" for i in range(1, 23, 2)]
group_names = [f"DRE18_{i}" for i in range(0, 16)]



group_names = ["DRE17_13", "DRE18_14", "DIVBO03_0"]
group_names = ["DIVBO03_1", "DIVBO01_1"]
group_names = ["DIVBO03_1", "DRE15_7"]
group_names = [f"DRE21_{i}" for i in range(0, 11)]

until_iteration = 100

for group_name in group_names:
    print("w")
    # Fetch runs
    runs = api.runs(
        f"{user_name}/{project_name}",
        filters={"$and": [{"group": group_name}, {"state": "finished"}]},
    )

    for run in runs:
        history = run.history(
            keys=["incumbent (norm)", "searcher_iteration"], pandas=False
        )
        max_num_pipelines = run.config["max_num_pipelines"]
        dataset_id = run.config["dataset_id"]
        seed = str(run.config["seed"])

        if history:
            incumbent_values = [record["incumbent (norm)"] for record in history]
            iteration_values = [record["searcher_iteration"] for record in history]

            if len(incumbent_values) >= until_iteration:
                results[max_num_pipelines][group_name][str(dataset_id) + seed] = incumbent_values[:until_iteration]
num_datasets = 6
num_groups = len(group_names)
max_num_pipelines_values = [1, 2, 4, 6, 8, 10]
max_num_pipelines_values = [5]
#group_names = ["RS00", "DRE06", "LEO00"]
#group_names = ["RS01_1", "DRE10_1", "DRE11_1", "LEO02_1", "DIVBO01_1"]
dataset_ids = ['micro_set0_RESISC_v1', 'micro_set1_MD_5_BIS_v1', 'micro_set2_BTS_v1', 'micro_set1_INS_2_v1', 'micro_set2_PRT_v1', 'micro_set2_INS_v1']
#dataset_ids = ['credit-approval', 'PhishingWebsites', 'ozone-level-8hr', 'pc1', 'cmc']
dataset_id = list(range(num_datasets))
seeds = [0]
dataset_seed_ids = list(itertools.product(dataset_ids, seeds))
dataset_seed_ids = [a+str(b) for a, b in dataset_seed_ids]
for max_num_pipelines in max_num_pipelines_values:
    results_matrix = []

    for dataset_seed_id in dataset_seed_ids:
        temp_results = []
        omit_dataset = False
        for group_name in group_names:
            if dataset_seed_id in results[max_num_pipelines][group_name].keys():
                incumbent_values = results[max_num_pipelines][group_name][dataset_seed_id]
                if len(incumbent_values) >= until_iteration:
                    temp_results.append(incumbent_values[:until_iteration])
                else:
                    print(
                        "Problem with dataset_id:", dataset_seed_id, "in group_name:", group_name
                    )
           
            else:
                omit_dataset = True
                print(
                    "Problem with dataset_id:", dataset_seed_id, "in group_name:", group_name
                )

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
    plt.savefig(
        os.path.join(current_file_path, output_folder, f"rank15_{max_num_pipelines}.png")
    )

    plt.figure()
    plt.plot(regret.T)
    plt.legend(group_names)
    plt.savefig(
        os.path.join(current_file_path, output_folder, f"regret15_{max_num_pipelines}.png")
    )


group_name = "DRE07"

results_matrix = []

max_num_pipelines_values = [1, 2, 4, 6, 8, 10]
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
plt.savefig(
    os.path.join(
        current_file_path, output_folder, f"{group_name}_max_num_pipelines_rank.png"
    )
)

plt.figure()
plt.plot(regret.T)
plt.legend(max_num_pipelines_values)
plt.savefig(
    os.path.join(
        current_file_path, output_folder, f"{group_name}_max_num_pipelines_regret.png"
    )
)


group_name = "DRE02"
results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
runs = api.runs(
    f"{user_name}/{project_name}",
    filters={"$and": [{"group": group_name}, {"state": "finished"}]},
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

for dataset_id in [0, 2, 3, 5]:
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
plt.savefig(
    os.path.join(current_file_path, output_folder, f"{group_name}_comparison.png")
)

plt.figure()
plt.plot(regret.T)
plt.legend(config_ids)
plt.savefig(
    os.path.join(current_file_path, output_folder, f"{group_name}_comparison.png")
)
