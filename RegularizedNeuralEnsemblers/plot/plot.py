from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .utils import calc_mean_and_sem, fetch_results_and_configs

BASE_FONT_SIZE = 14


def _plot_single_subplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_axis: str,
    plot_function: Callable,
    dataset_name: str | None = None,
    x_range: tuple | None = None,
    log_x: bool = False,
    log_y: bool = False,
):
    plot_function(ax, data, dataset_name=dataset_name, y_axis=y_axis)
    # Set tight autoscaling
    ax.autoscale(enable=True, axis="x", tight=True)
    if x_range:
        ax.set_xlim(x_range)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def plot_general(
    results: pd.DataFrame,
    output_dir: Path,
    plot_function: Callable,
    subplot_shape: tuple | None = None,
    x_range: tuple | None = None,
    log_x: bool = False,
    log_y: bool = False,
    plot_name: str = "plot",
    extension: str = "png",
    dpi: int = 100,
    y_axis: str = "incumbent",
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for metadataset_name in results.index.get_level_values("metadataset_name").unique():
        metadataset_data = results.xs(metadataset_name, level="metadataset_name", axis=0)

        if subplot_shape:
            nrows, ncols = subplot_shape
            fig, axs = plt.subplots(nrows, ncols, figsize=(18, 18))
            fig.subplots_adjust(hspace=0.5)
            axs = axs.ravel()
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            axs = [ax]  # make it a list for consistency

        fig.suptitle(f"Metadataset: {metadataset_name}", fontsize=BASE_FONT_SIZE)

        if subplot_shape:
            num_visible_plots = len(
                metadataset_data.index.get_level_values("dataset_name").unique()
            )
            last_visible_row = (num_visible_plots - 1) // ncols

            for ax_idx, (dataset_name, dataset_data) in enumerate(
                metadataset_data.groupby("dataset_name")
            ):
                _plot_single_subplot(
                    ax=axs[ax_idx],
                    data=dataset_data,
                    dataset_name=dataset_name,
                    y_axis=y_axis,
                    plot_function=plot_function,
                    x_range=x_range,
                    log_x=log_x,
                    log_y=log_y,
                )
                if ax_idx >= len(axs) - 1:
                    break

            # Hide the remaining subplots
            for ax_idx in range(num_visible_plots, nrows * ncols):
                axs[ax_idx].set_visible(False)

            bbox_to_anchor = (0.5, 0.25 * (last_visible_row + 1))

        else:
            _plot_single_subplot(
                ax=ax,
                data=metadataset_data,
                y_axis=y_axis,
                plot_function=plot_function,
                x_range=x_range,
                log_x=log_x,
                log_y=log_y,
            )

            bbox_to_anchor = (0.5, 0.05)

        # Add legend to the plot
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=4,
            bbox_transform=fig.transFigure,
            fontsize=BASE_FONT_SIZE - 2,
        )

        file_path = os.path.join(
            output_dir, f"{plot_name}_{metadataset_name}.{extension}"
        )
        plt.savefig(file_path, bbox_inches="tight", dpi=dpi)
        print(f"Plot saved to {file_path}")
        plt.close(fig)


def regret_plot_function(
    ax, df: pd.DataFrame, dataset_name: str, y_axis: str = "incumbent"
):
    # Group by 'group_name' and 'dataset_name', and then apply the custom function
    aggregated_data = df.groupby(["group_name", "dataset_name"]).apply(
        lambda group_df: calc_mean_and_sem(group_df, y_axis)
    )

    for (group_name, _), (mean_values, sem_values) in aggregated_data.items():
        ax.plot(mean_values, label=f"Group: {group_name}")
        ax.fill_between(
            range(len(mean_values)),
            mean_values - sem_values,
            mean_values + sem_values,
            alpha=0.3,
        )

    ax.set_title(f"Dataset: {dataset_name}", fontsize=BASE_FONT_SIZE)
    ax.set_xlabel("Iteration", fontsize=BASE_FONT_SIZE - 2)
    ax.set_ylabel(y_axis, fontsize=BASE_FONT_SIZE - 2)


def aggregated_regret_plot_function(
    ax,
    df: pd.DataFrame,
    y_axis: str = "incumbent",
    **kwargs,  # pylint: disable=unused-argument
):
    all_datasets = df.index.get_level_values("dataset_name").unique()

    # Placeholder for the final aggregated DataFrame for all datasets
    all_regrets_dfs = []

    # Iterate over each dataset
    for dataset_name in all_datasets:
        # Filter data to match the current dataset
        dataset_df = df.loc[
            (slice(None), dataset_name, slice(None)), :
        ]  # Using slicers to select the appropriate rows

        # Group by 'group_name' and then apply the custom function
        aggregated_data = dataset_df.groupby(["group_name"]).apply(
            lambda group_df: calc_mean_and_sem(group_df, y_axis)
        )

        # Convert aggregated_data (which has mean and SEM) to a DataFrame with MultiIndex columns
        stats_df = pd.DataFrame(
            aggregated_data.tolist(), columns=["mean", "sem"], index=aggregated_data.index
        )
        stats_df.columns = pd.MultiIndex.from_product([[dataset_name], stats_df.columns])

        all_regrets_dfs.append(stats_df)

    # Concatenate all DataFrames to get the final DataFrame with all datasets
    all_regrets_df = pd.concat(all_regrets_dfs, axis=1)

    # Calculate mean and SEM across datasets
    mean_df = all_regrets_df.xs("mean", level=1, axis=1)
    mean_across_datasets = mean_df.apply(
        lambda row: np.mean(np.array(row.tolist()), axis=0), axis=1
    )

    sem_df = all_regrets_df.xs("sem", level=1, axis=1)
    sem_across_datasets = sem_df.apply(
        lambda row: np.mean(np.array(row.tolist()), axis=0), axis=1
    )

    # Assuming that each list of means or sems has the same length
    x = range(len(mean_across_datasets.iloc[0]))

    for group_name, mean_values in mean_across_datasets.iteritems():
        ax.plot(x, mean_values, label=f"Group: {group_name}")
        ax.fill_between(
            x,
            mean_values - sem_across_datasets[group_name],
            mean_values + sem_across_datasets[group_name],
            alpha=0.3,
        )

    ax.set_xlabel("Iteration", fontsize=BASE_FONT_SIZE - 2)
    ax.set_ylabel(y_axis, fontsize=BASE_FONT_SIZE - 2)


def rank_plot_function(
    ax,
    df: pd.DataFrame,
    dataset_name: str,
    y_axis: str = "incumbent",
    **kwargs,  # pylint: disable=unused-argument
):
    # Filter data to match the current dataset
    dataset_df = df.loc[
        (slice(None), dataset_name, slice(None)), :
    ]  # Using slicers to select the appropriate rows

    # Group by 'group_name' and then apply the custom function
    aggregated_data = dataset_df.groupby(["group_name"]).apply(
        lambda group_df: calc_mean_and_sem(group_df, y_axis)
    )

    # Convert aggregated_data (which has mean and SEM) to a DataFrame
    mean_values_df = pd.DataFrame(
        {group: data[0] for group, data in aggregated_data.items()}
    )
    # For each iteration, rank the groups
    ranks = mean_values_df.apply(rankdata, axis=1)
    rank_df = pd.DataFrame(ranks.tolist(), columns=mean_values_df.columns)

    # Plot the rank data
    for group in rank_df.columns:
        ax.plot(rank_df[group], label=f"Group: {group}")

    ax.set_title(f"Dataset: {dataset_name}", fontsize=BASE_FONT_SIZE)
    ax.set_xlabel("Iteration", fontsize=BASE_FONT_SIZE - 2)
    ax.set_ylabel("Rank", fontsize=BASE_FONT_SIZE - 2)


def aggregated_rank_plot_function(
    ax,
    df: pd.DataFrame,
    y_axis: str = "incumbent",
    **kwargs,  # pylint: disable=unused-argument
):
    # all_groups = df.index.get_level_values('group_name').unique()
    all_datasets = df.index.get_level_values("dataset_name").unique()

    # Placeholder to store rank data for plotting, using a dict for ease of concatenation later
    all_ranks = {}

    # Iterate over each dataset
    for dataset_name in all_datasets:
        # Filter data to match the current dataset
        dataset_df = df.loc[
            (slice(None), dataset_name, slice(None)), :
        ]  # Using slicers to select the appropriate rows

        # Group by 'group_name' and then apply the custom function
        aggregated_data = dataset_df.groupby(["group_name"]).apply(
            lambda group_df: calc_mean_and_sem(group_df, y_axis)
        )

        # Convert aggregated_data (which has mean and SEM) to a DataFrame
        mean_values_df = pd.DataFrame(
            {group: data[0] for group, data in aggregated_data.items()}
        )
        # For each iteration, rank the groups
        ranks = mean_values_df.apply(rankdata, axis=1)
        rank_df = pd.DataFrame(ranks.tolist(), columns=mean_values_df.columns)

        # Store ranks in our dict
        all_ranks[dataset_name] = rank_df

    # Concatenate all DataFrames along columns axis
    all_ranks_df = pd.concat(all_ranks.values(), keys=all_ranks.keys(), axis=1)

    # Calculate average ranks across datasets
    mean_ranks_df = all_ranks_df.mean(level=1, axis=1)

    # Plot the rank data
    for group in mean_ranks_df.columns:
        ax.plot(mean_ranks_df[group], label=f"Group: {group}")

    ax.set_xlabel("Iteration", fontsize=BASE_FONT_SIZE - 2)
    ax.set_ylabel("Rank", fontsize=BASE_FONT_SIZE - 2)


def plot(
    user_name: str,
    project_name: str,
    group_names: list[str],
    ######################################
    output_dir: Path,
    ######################################
    plot_type: str = "regret",
    unnormalize: bool = False,
    per_dataset: bool = False,
    ######################################
    x_axis: str = "searcher_iteration",
    y_axis: str = "incumbent",
    ######################################
    x_range: tuple | None = None,
    log_x: bool = False,
    log_y: bool = False,
    #######################################
    plot_name: str = "plot",
    extension: str = "png",
    dpi: int = 100,
    #######################################
) -> None:
    plot_name = f"{plot_name}_{plot_type}"
    if not unnormalize:
        y_axis = f"{y_axis} (norm)"
    plot_name = f"{plot_name}_aggregated" if not per_dataset else plot_name
    plot_name = f"{plot_name}_normalized" if not unnormalize else plot_name

    # Create output directory, using path
    output_dir.mkdir(parents=True, exist_ok=True)
    # Fetch results and configs
    results, _ = fetch_results_and_configs(
        user_name=user_name,
        project_name=project_name,
        group_names=group_names,
        x_axis=x_axis,
        y_axis=y_axis,
    )

    plotting_function = partial(
        plot_general,
        results=results,
        output_dir=output_dir,
        x_range=x_range,
        log_x=log_x,
        log_y=log_y,
        plot_name=plot_name,
        extension=extension,
        dpi=dpi,
        y_axis=y_axis,
    )

    # Plot results
    if plot_type == "regret":
        if per_dataset:
            plotting_function(
                plot_function=regret_plot_function,
                subplot_shape=(4, 3),
            )
        else:
            plotting_function(
                plot_function=aggregated_regret_plot_function,
                subplot_shape=None,
            )
    elif plot_type == "rank":
        if per_dataset:
            plotting_function(
                plot_function=rank_plot_function,
                subplot_shape=(4, 3),
            )
        else:
            plotting_function(
                plot_function=aggregated_rank_plot_function,
                subplot_shape=None,
            )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
