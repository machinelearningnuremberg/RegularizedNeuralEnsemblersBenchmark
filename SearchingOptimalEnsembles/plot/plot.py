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
    dataset_name: str,
    y_axis: str,
    plot_function: Callable,
    x_range: tuple | None = None,
    log_x: bool = False,
    log_y: bool = False,
):
    plot_function(ax, data, dataset_name=dataset_name, y_axis=y_axis)
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
            dataset_name, dataset_data = next(
                iter(metadataset_data.groupby("dataset_name"))
            )
            _plot_single_subplot(
                ax=ax,
                data=dataset_data,
                dataset_name=dataset_name,
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
    # # Filter data for this particular dataset
    # df = df.xs(dataset_name, level="dataset_name")

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
    # Group by 'group_name' and then apply the custom function
    aggregated_data = df.groupby(["group_name"]).apply(
        lambda group_df: calc_mean_and_sem(group_df, y_axis)
    )

    for group_name, (mean_values, sem_values) in aggregated_data.items():
        ax.plot(mean_values, label=f"Group: {group_name}")
        ax.fill_between(
            range(len(mean_values)),
            mean_values - sem_values,
            mean_values + sem_values,
            alpha=0.3,
        )

    ax.set_xlabel("Iteration", fontsize=BASE_FONT_SIZE - 2)
    ax.set_ylabel(y_axis, fontsize=BASE_FONT_SIZE - 2)


def rank_plot_function(
    ax, df: pd.DataFrame, dataset_name: str, y_axis: str = "incumbent"
):
    # Filter data for this particular dataset
    df = df.xs(dataset_name, level="dataset_name")

    # Group by 'group_name' and then apply your custom function to calculate mean and SEM
    aggregated_data = df.groupby("group_name").apply(
        lambda group_df: calc_mean_and_sem(group_df, y_axis)
    )

    # Find the number of iterations
    first_group_data = aggregated_data.iloc[0]
    num_iterations = len(first_group_data[0])

    # Initialize a dictionary to store ranks for each group at each iteration
    rank_over_time: dict = {group_name: [] for group_name in aggregated_data.index}

    # Loop over each iteration and calculate ranks
    for i in range(num_iterations):
        iteration_means = [
            data[0][i] if i < len(data[0]) else np.nan for data in aggregated_data
        ]
        ranks = rankdata(iteration_means, method="min")  # Ranks at this iteration
        for j, group_name in enumerate(aggregated_data.index):
            rank_over_time[group_name].append(ranks[j])

    # Plotting ranks over time
    for group_name, ranks in rank_over_time.items():
        ax.plot(ranks, label=f"Group: {group_name}")

    ax.set_title(f"Dataset: {dataset_name}", fontsize=BASE_FONT_SIZE)
    ax.set_xlabel("Iteration", fontsize=BASE_FONT_SIZE - 2)
    ax.set_ylabel("Rank", fontsize=BASE_FONT_SIZE - 2)


def aggregated_rank_plot_function(
    ax,
    df: pd.DataFrame,
    y_axis: str = "incumbent",
    **kwargs,  # pylint: disable=unused-argument
):
    # Group by 'group_name' and apply custom function to calculate mean and SEM
    aggregated_data = df.groupby(level="group_name").apply(
        lambda group_df: calc_mean_and_sem(group_df, y_axis)
    )

    # Find the number of iterations
    first_group_data = aggregated_data.iloc[0]
    num_iterations = len(first_group_data[0])

    # Initialize a dictionary to store ranks for each group at each iteration
    rank_over_time: dict = {group_name: [] for group_name in aggregated_data.index}

    # Loop over each iteration to calculate ranks
    for i in range(num_iterations):
        iteration_means = [
            data[0][i] if i < len(data[0]) else np.nan for data in aggregated_data
        ]
        ranks = rankdata(iteration_means, method="min")
        for j, group_name in enumerate(aggregated_data.index):
            rank_over_time[group_name].append(ranks[j])

    # Plotting ranks over time
    for group_name, ranks in rank_over_time.items():
        ax.plot(ranks, label=f"Group: {group_name}")

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
