# type: ignore
# pylint: skip-file

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

current_dir = os.path.dirname(os.path.realpath(__file__))


def get_dataset_names():
    return os.listdir(os.path.join(current_dir, "..", "results", "EO1"))


def read_results(results_path, dataset_names):
    results = {"aggregated_val_perf": {}, "aggregated_test_perf": {}}

    for dataset_name in dataset_names:
        # read json
        # read json file
        try:
            with open(os.path.join(results_path, dataset_name, "results.json")) as f:
                temp_results = eval(json.load(f))

            # results["aggregated_val_perf"].append(results[dataset_name]["val_perf"])
            # results["aggregated_test_perf"].append(results[dataset_name]["test_perf"])
            results["aggregated_val_perf"][dataset_name] = temp_results["val_perf"]
            results["aggregated_test_perf"][dataset_name] = temp_results["test_perf"]
        except Exception as e:
            print(e)
            continue

    return results


def gather_results(experiments, results_path):
    unorganized_results = {"aggregated_val_perf": {}, "aggregated_test_perf": {}}
    dataset_names = get_dataset_names()
    # gather results
    for experiment_id in experiments:
        try:
            temp_results = read_results(
                os.path.join(results_path, experiment_id), dataset_names
            )
        except Exception as e:
            print(e)
            continue

        unorganized_results["aggregated_val_perf"][experiment_id] = temp_results[
            "aggregated_val_perf"
        ]
        unorganized_results["aggregated_test_perf"][experiment_id] = temp_results[
            "aggregated_test_perf"
        ]

    results = {"aggregated_val_perf": [], "aggregated_test_perf": []}

    for dataset in dataset_names:
        try:
            temp_results_val = []
            temp_results_test = []
            for experiment_id in experiments:
                temp_results_val.append(
                    unorganized_results["aggregated_val_perf"][experiment_id][dataset]
                )
                temp_results_test.append(
                    unorganized_results["aggregated_test_perf"][experiment_id][dataset]
                )
            results["aggregated_val_perf"].append(temp_results_val)
            results["aggregated_test_perf"].append(temp_results_test)
        except Exception as e:
            print(
                "Error in dataset: ",
                dataset,
                " , experiment:",
                experiment_id,
                " , error: ",
                e,
            )
            continue
    return results


def plot_results(
    experiment_ids, experiment_names, add_confidence_interval=False, save_path=None
):
    results_path = os.path.join(current_dir, "..", "results")

    # create subplot with two figures
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # gather results
    results = gather_results(experiment_ids, results_path)

    regret_val = np.array(results["aggregated_val_perf"])
    regret_test = np.array(results["aggregated_test_perf"])

    rank_val = rankdata(regret_val, axis=1)
    rank_test = rankdata(regret_test, axis=1)

    regret_val_mean = regret_val.mean(0)
    regret_test_mean = regret_test.mean(0)
    val_results_rank_mean = rank_val.mean(0)
    test_results_rank_mean = rank_test.mean(0)

    n_samples = regret_val.shape[0]
    factor = 1.96 / np.sqrt(n_samples)

    regret_val_std = regret_val.std(0) * factor
    regret_test_std = regret_test.std(0) * factor
    val_results_std_rank = rank_val.std(0) * factor
    test_results_std_rank = rank_test.std(0) * factor

    all_results = [
        (regret_val_mean, regret_val_std),
        (regret_test_mean, regret_test_std),
        (val_results_rank_mean, val_results_std_rank),
        (test_results_rank_mean, test_results_std_rank),
    ]

    # all_results = [(regret_val_mean, regret_val_std), (regret_test_mean, regret_test_std),
    #               (regret_val_mean, regret_val_std), (regret_test_mean, regret_test_std)]

    for i, (means, stds) in enumerate(all_results):
        i1 = i // 2
        i2 = i % 2
        for j in range(len(experiment_names)):
            mean = means[j]
            std = stds[j]
            axs[i1, i2].plot(mean, label=experiment_names[j])
            if add_confidence_interval:
                axs[i1, i2].fill_between(
                    np.arange(len(mean)), mean - std, mean + std, alpha=0.2
                )

    axs[0, 0].set_title("Validation")
    axs[0, 1].set_title("Test")

    axs[1, 0].set_ylabel("Normalized Regret")
    axs[1, 1].set_ylabel("Rank")

    axs[1, 0].set_xlabel("Observed Pipelines")
    axs[1, 1].set_xlabel("Observed Pipelines")

    plt.legend(experiment_names)
    # plt.show()

    if save_path is not None:
        plt.savefig(os.path.join(current_dir, "..", "plots", save_path))


experiment_names = [
    "EO",
    "MetaEnsembleLearning",
    "BO",
    "Oracle",
    "NoMetaLearned1",
    "NoMetaLearned2",
    "T1",
    "T2",
]
experiment_ids = ["EO1", "TBOE1", "EO2", "EO_oracle", "TBOE3", "TBOE2", "TBOE7", "TBOE8"]

experiment_names = ["EO", "BO", "Oracle", "T2"]
experiment_ids = ["EO1", "EO2", "EO_oracle", "TBOE8"]

plot_results(experiment_ids, experiment_names, save_path="results.png")
