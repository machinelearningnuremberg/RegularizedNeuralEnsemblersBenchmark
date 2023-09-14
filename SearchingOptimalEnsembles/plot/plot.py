from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

BASE_FONT_SIZE = 14


def fetch_results(
    user_name: str,
    project_name: str,
    group_name: str,
    metadatasets: list[str] | None = None,
    algorithms: list[str] | None = None,
    x_axis: str = "searcher_iteration",
    y_axis: str = "incumbent (norm)",
):
    api = wandb.Api()

    # Dictionary to hold results
    results: defaultdict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Fetch runs
    runs = api.runs(
        f"{user_name}/{project_name}",
        filters={"$and": [{"group": group_name}, {"state": "finished"}]},
    )

    # Algorithm filter, algorithms are passed as searcher_name + surrogate_name, joined by "_"
    # e.g. "bo_dkl", let's get all possible searchers and surrogates
    if algorithms is not None:
        searcher_names = set()
        surrogate_names = set()
        for algorithm in algorithms:
            searcher_name, surrogate_name = algorithm.split("_")
            searcher_names.add(searcher_name)
            surrogate_names.add(surrogate_name)

    # Iterate over runs
    for run in runs:
        # Fetch the run name
        full_run_name = run.name
        identifier = full_run_name.rsplit("_", 1)[
            0
        ]  # Remove the last digit to get the identifier

        # searcher_name, metadataset_name, surrogate_name = run.group.split("_")
        metadataset_name = [
            tag.split("=")[1] for tag in run.tags if "metadataset" in tag
        ][0]
        if metadatasets is not None and metadataset_name not in metadatasets:
            continue
        searcher_name = [tag.split("=")[1] for tag in run.tags if "searcher" in tag][0]
        if algorithms is not None and searcher_name not in searcher_names:
            continue
        surrogate_name = [tag.split("=")[1] for tag in run.tags if "surrogate" in tag][0]
        if algorithms is not None and surrogate_name not in surrogate_names:
            continue
        dataset_name = [tag.split("=")[1] for tag in run.tags if "dataset" in tag][0]
        # seed = [tag.split("=")[1] for tag in run.tags if "seed" in tag][0]
        meta_num_epochs = [
            tag.split("=")[1] for tag in run.tags if "meta_num_epochs" in tag
        ][0]
        max_num_pipelines = [
            tag.split("=")[1] for tag in run.tags if "max_num_pipelines" in tag
        ][0]

        # Include the identifier as part of the key
        key = (
            identifier,
            searcher_name,
            surrogate_name,
            meta_num_epochs,
            max_num_pipelines,
        )

        history = run.history(keys=[y_axis, x_axis], pandas=False)
        if history:
            # key = (searcher_name, surrogate_name, meta_num_epochs, max_num_pipelines)
            incumbent_values = [record[y_axis] for record in history]
            iteration_values = [record[x_axis] for record in history]

            results[metadataset_name][dataset_name][key].append(
                (iteration_values, incumbent_values)
            )

    return results


def plot_regret(
    results: defaultdict,
    output_dir: Path,
    x_range: tuple | None = None,
    log_x: bool = False,
    log_y: bool = False,
    plot_name: str = "regret",
    extension: str = "png",
    dpi: int = 100,
) -> None:
    # Plot results
    for metadataset_name, metadataset_results in results.items():
        # num_datasets = len(metadataset_results)
        fig, axs = plt.subplots(4, 3, figsize=(18, 18))  # 4 rows, 3 columns
        fig.suptitle(
            f"Metadataset: {metadataset_name}", fontsize=BASE_FONT_SIZE + 4, y=0.92
        )

        # Reduce space between subplots and legend
        plt.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.15)

        # Hide empty subplots
        for ax_idx in range(
            len(metadataset_results), 12
        ):  # 4 rows x 3 columns = 12 subplots
            row, col = divmod(ax_idx, 3)
            axs[row, col].set_visible(False)

        handles, labels = None, None
        for ax_idx, (dataset_name, dataset_results) in enumerate(
            metadataset_results.items()
        ):
            row, col = divmod(ax_idx, 3)  # 3 columns
            ax = axs[row, col]

            for key in dataset_results.keys():
                all_iterations = []
                all_incumbents = []

                for iteration_values, incumbent_values in dataset_results[key]:
                    all_iterations.append(
                        iteration_values
                    )  # Now we append the list instead of extending it
                    all_incumbents.append(incumbent_values)  # Same here

                # Convert to a numpy array and average over the first axis (assumed to be over seeds)
                all_incumbents_array = np.array(all_incumbents)
                mean_incumbents = np.mean(all_incumbents_array, axis=0)

                # Calculate standard deviation and standard error
                std_dev_incumbents = np.std(all_incumbents_array, axis=0)
                num_seeds = all_incumbents_array.shape[0]
                std_err_incumbents = std_dev_incumbents / np.sqrt(num_seeds)

                # Make sure all lists have the same length before plotting
                unique_iterations = sorted(
                    set(all_iterations[0])
                )  # Assuming all lists of iterations are identical

                # if len(unique_iterations) != len(mean_incumbents):
                #     raise ValueError(f"Mismatched dimensions: {len(unique_iterations)} vs {len(mean_incumbents)}")

                (_,) = ax.plot(unique_iterations, mean_incumbents, label=str(key))
                ax.fill_between(
                    unique_iterations,
                    mean_incumbents - std_err_incumbents,
                    mean_incumbents + std_err_incumbents,
                    alpha=0.3,
                )

                # Set the x-axis range
                if x_range is not None:
                    ax.set_xlim(x_range)

                # Set the x-axis to log scale
                if log_x:
                    ax.set_xscale("log")

                # Set the y-axis to log scale
                if log_y:
                    ax.set_yscale("log")

            if col == 0:
                ax.set_ylabel("Incumbent", fontsize=BASE_FONT_SIZE)

            if row == 1:
                ax.set_xlabel("Iteration", fontsize=BASE_FONT_SIZE)

            if ax_idx == 0:
                handles, labels = ax.get_legend_handles_labels()

            ax.set_title(f"Dataset: {dataset_name}", fontsize=BASE_FONT_SIZE)

        # Common legend below the entire figure
        # Create a new axes for the legend at the bottom of the figure
        legend_ax = fig.add_axes([0.1, 0.45, 0.8, 0.05])  # [left, bottom, width, height]

        # Add the legend to the new axes
        legend_ax.legend(handles, labels, loc="center", ncol=4, fontsize=BASE_FONT_SIZE)

        # Hide the axes
        legend_ax.axis("off")

        # Save to disk
        file_path = os.path.join(output_dir, f"{plot_name}.{extension}")
        plt.savefig(file_path, bbox_inches="tight", dpi=dpi)
        print(f"Plot saved to {file_path}")

        # Close the figure to free up memory
        plt.close(fig)


def plot_aggregated_regret(
    results: defaultdict,
    output_dir: Path,
    x_range: tuple | None = None,
    log_x: bool = False,
    log_y: bool = False,
    plot_name: str = "aggregated_regret",
    extension: str = "png",
    dpi: int = 100,
    force_name: bool = False,
) -> None:
    for metadataset_name, metadataset_results in results.items():
        fig, ax = plt.subplots(figsize=(10, 8))  # Single plot per metadataset
        fig.suptitle(
            f"Metadataset: {metadataset_name}",
            fontsize=BASE_FONT_SIZE + 4,
        )

        aggregated_data = defaultdict(list)

        for _, dataset_configs in metadataset_results.items():
            for config, runs in dataset_configs.items():
                identifier = config[0]
                searcher_name = config[1]
                surrogate_name = config[2]
                num_meta_epochs = config[3]
                max_num_pipelines = config[4]
                label = (
                    identifier
                    if force_name
                    else f"{searcher_name}_{surrogate_name}_{num_meta_epochs}_{max_num_pipelines}"
                )

                for _, incumbent_values in runs:
                    # Append incumbent values of each dataset to be aggregated later
                    aggregated_data[label].append(incumbent_values)

        for label, all_incumbents in aggregated_data.items():
            # Convert to a numpy array and average over the first two axes (first over seeds, then over datasets)
            all_incumbents_array = np.array(all_incumbents)

            # Ensure it has three dimensions
            if len(all_incumbents_array.shape) == 2:
                all_incumbents_array = np.expand_dims(all_incumbents_array, axis=0)

            mean_incumbents = np.mean(all_incumbents_array, axis=(0, 1))

            # Calculate standard deviation and standard error
            std_dev_incumbents = np.std(np.mean(all_incumbents_array, axis=1), axis=0)
            num_seeds = all_incumbents_array.shape[0]
            std_err_incumbents = std_dev_incumbents / np.sqrt(num_seeds)

            # Assuming all lists of iterations are identical across datasets and runs
            unique_iterations = np.array(range(len(mean_incumbents)))

            # Plotting
            ax.plot(unique_iterations, mean_incumbents, label=label)
            ax.fill_between(
                unique_iterations,
                mean_incumbents - std_err_incumbents,
                mean_incumbents + std_err_incumbents,
                alpha=0.3,
            )

        # Common settings
        ax.set_ylabel("Aggregated Normalized Regret", fontsize=BASE_FONT_SIZE)
        ax.set_xlabel("Iteration", fontsize=BASE_FONT_SIZE)

        if x_range is not None:
            ax.set_xlim(x_range)

        if log_x:
            ax.set_xscale("log")

        if log_y:
            ax.set_yscale("log")

        # If force_name is True, save legend separately
        if force_name:
            fig_legend = plt.figure(figsize=(3, 4))
            handles, labels = ax.get_legend_handles_labels()
            plt.figlegend(handles, labels, loc="center")
            legend_file_path = os.path.join(
                output_dir, f"{plot_name}_{metadataset_name}_legend.{extension}"
            )
            fig_legend.savefig(legend_file_path, bbox_inches="tight", dpi=dpi)
            print(f"Legend saved to {legend_file_path}")
            plt.close(fig_legend)
        else:
            ax.legend(loc="best", fontsize=BASE_FONT_SIZE)

        # Save to disk
        file_path = os.path.join(
            output_dir, f"{plot_name}_{metadataset_name}.{extension}"
        )
        plt.savefig(file_path, bbox_inches="tight", dpi=dpi)
        print(f"Plot saved to {file_path}")

        # Close the figure to free up memory
        plt.close(fig)


def plot(
    user_name: str,
    project_name: str,
    group_name: str,
    output_dir: Path,
    metadatasets: list[str] | None = None,
    algorithms: list[str] | None = None,
    plot_type: str = "regret",
    normalize: bool = True,
    x_range: tuple | None = None,
    log_x: bool = False,
    log_y: bool = False,
    plot_name: str = "plot",
    extension: str = "png",
    dpi: int = 100,
) -> None:
    x_axis = "searcher_iteration"
    y_axis = "incumbent (norm)" if normalize else "incumbent"
    plot_name = f"{plot_name}_normalized" if normalize else plot_name

    # Create output directory, using path
    output_dir.mkdir(parents=True, exist_ok=True)
    # Fetch results
    results = fetch_results(
        user_name=user_name,
        project_name=project_name,
        group_name=group_name,
        metadatasets=metadatasets,
        algorithms=algorithms,
        x_axis=x_axis,
        y_axis=y_axis,
    )

    # Plot results§
    if plot_type == "regret":
        plot_regret(
            results=results,
            output_dir=output_dir,
            x_range=x_range,
            log_x=log_x,
            log_y=log_y,
            plot_name=plot_name,
            extension=extension,
            dpi=dpi,
        )
    elif plot_type == "aggregated_regret":
        plot_aggregated_regret(
            results=results,
            output_dir=output_dir,
            x_range=x_range,
            log_x=log_x,
            log_y=log_y,
            plot_name=plot_name,
            extension=extension,
            dpi=dpi,
            force_name=True,
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
