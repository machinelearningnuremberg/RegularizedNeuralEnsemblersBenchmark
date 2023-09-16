from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import wandb


def calc_mean_and_sem(group_df, y_axis):
    # Here, group_df will be the subset of the original df that corresponds to each (group_name, dataset_name)
    y_values_list = [nested_df[y_axis].values for nested_df in group_df["history"]]

    # Stack them into a NumPy array for easy mean and SEM computation
    y_values_array = np.stack(y_values_list, axis=0)

    mean_values = np.mean(y_values_array, axis=0)
    sem_values = np.std(y_values_array, axis=0) / np.sqrt(y_values_array.shape[0])

    return mean_values, sem_values


def fetch_results_and_configs(
    user_name: str,
    project_name: str,
    group_names: list[str],
    x_axis: str = "searcher_iteration",
    y_axis: str = "incumbent (norm)",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches experiment results for a list of groups from wandb and returns them as a multi-indexed Pandas DataFrame,
    along with the configurations for each group in another DataFrame.

    Args:
        user_name (str): The username of the wandb account.
        project_name (str): The name of the wandb project.
        group_names (List[str]): List of groups to fetch data for.
        x_axis (str, optional): The name of the x-axis variable in the run history. Defaults to "searcher_iteration".
        y_axis (str, optional): The name of the y-axis variable in the run history. Defaults to "incumbent (norm)".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - results_df: A multi-indexed Pandas DataFrame containing the fetched results. The DataFrame is indexed by
              ['group_name', 'metadataset_name', 'dataset_name', 'seed'] and contains columns specified by x_axis and y_axis.
            - configs_df: A Pandas DataFrame containing the group configurations. The DataFrame is indexed by ['group_name']
              and contains configuration parameters as columns.

    Raises:
        ValueError: If the run configuration within a group is inconsistent.

    Example:
        To query results from a specific group and metadataset, you can use:
        >>> results_df.loc[('group_name', 'metadataset_name')]
        To query the configuration of a specific group, you can use:
        >>> configs_df.loc['group_name']
    """

    api = wandb.Api()

    all_results = []
    all_configs = []

    for group_name in group_names:
        group_results, group_config = _fetch_group_results_and_config(
            api=api,
            user_name=user_name,
            project_name=project_name,
            group_name=group_name,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        all_results += group_results
        group_config[
            "group_name"
        ] = group_name  # Add group_name to each config dict for DataFrame conversion
        all_configs.append(group_config)

    # Setting multi-index
    results_df = pd.DataFrame(all_results)
    results_df.set_index(
        ["group_name", "metadataset_name", "dataset_name", "seed"], inplace=True
    )

    configs_df = pd.DataFrame(all_configs).set_index("group_name")

    return results_df, configs_df


def _fetch_group_results_and_config(
    api: wandb.Api,
    user_name: str,
    project_name: str,
    group_name: str,
    x_axis: str = "searcher_iteration",
    y_axis: str = "incumbent (norm)",
) -> tuple[list[dict[str, Any]], dict]:
    # Initialize results and configs
    group_results = []
    potential_configs = []

    # Fetch runs
    runs = api.runs(
        f"{user_name}/{project_name}",
        filters={"$and": [{"group": group_name}, {"state": "finished"}]},
    )

    # Collect potential reference configs and their corresponding runs
    run_records = []
    for run in runs:
        run_config = run.config.copy()
        metadataset_name = run_config.pop("metadataset_name", None)
        dataset_name = run_config.pop("dataset_id", None)
        seed = run_config.pop("seed", None)

        json_str = json.dumps(run_config, sort_keys=True)
        potential_configs.append(json_str)

        record = {
            "run": run,
            "run_config_json": json_str,
            "group_name": group_name,
            "metadataset_name": metadataset_name,
            "dataset_name": dataset_name,
            "seed": seed,
        }

        run_records.append(record)

    # Identify the most common config to be the reference
    common_config_json = Counter(potential_configs).most_common(1)[0][0]
    reference_config = json.loads(common_config_json)

    # Iterate over the collected runs and keep only those consistent with the reference config
    for record in run_records:
        if record["run_config_json"] != common_config_json:
            differences = {
                key: (
                    json.loads(record["run_config_json"]).get(key),
                    reference_config.get(key),
                )
                for key in set(json.loads(record["run_config_json"]))
                | set(reference_config)
                if json.loads(record["run_config_json"]).get(key)
                != reference_config.get(key)
            }
            error_msg = [
                f"Inconsistent configurations found for group {group_name}.",
                f"Inconsistent run: {record['run'].name}.",
                "Differences:",
            ]
            for key, (run_val, ref_val) in differences.items():
                error_msg.append(
                    f"  - {key}: Run value = {run_val}, Reference value = {ref_val}"
                )
            print("\n".join(error_msg))
            print("Skipping this run.\n")

            continue

        history_df = record["run"].history(keys=[y_axis, x_axis], pandas=True)
        if "_step" in history_df.columns:
            history_df.drop("_step", axis=1, inplace=True)

        record["history"] = history_df
        del record["run"]
        del record["run_config_json"]

        group_results.append(record)

    return group_results, reference_config
