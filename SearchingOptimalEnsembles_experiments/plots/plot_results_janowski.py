import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import os

base_font_size = 14
until_iteration = 100

api = wandb.Api()
user_name = "relea-research"
project_name = "SearchingOptimalEnsembles"
group_name = "debug"
output_folder = "./saved_plots"
os.makedirs(output_folder, exist_ok=True)

# Dictionary to hold results
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Fetch runs
runs = api.runs(f"{user_name}/{project_name}",
                filters={"$and": [{"group": group_name}, {"state": "finished"}]}
                )

# Iterate over runs
for run in runs:

    # searcher_name, metadataset_name, surrogate_name = run.group.split("_")
    metadataset_name = [tag.split("=")[1] for tag in run.tags if "metadataset" in tag][0]
    searcher_name = [tag.split("=")[1] for tag in run.tags if "searcher" in tag][0]
    surrogate_name = [tag.split("=")[1] for tag in run.tags if "surrogate" in tag][0]
    dataset_name = [tag.split("=")[1] for tag in run.tags if "dataset" in tag][0]
    seed = [tag.split("=")[1] for tag in run.tags if "seed" in tag][0]
    meta_num_epochs = [tag.split("=")[1] for tag in run.tags if "meta_num_epochs" in tag][0]
    max_num_pipelines = [tag.split("=")[1] for tag in run.tags if "max_num_pipelines" in tag][0]

    history = run.history(keys=["incumbent", "iteration"], pandas=False)
    if history:
        key = (searcher_name, surrogate_name, meta_num_epochs, max_num_pipelines)
        incumbent_values = [record["incumbent"] for record in history]
        iteration_values = [record["iteration"] for record in history]

        results[metadataset_name][dataset_name][key].append((iteration_values, incumbent_values))

# Plot results
for metadataset_name, metadataset_results in results.items():
    num_datasets = len(metadataset_results)
    fig, axs = plt.subplots(4, 3, figsize=(18, 18))  # 4 rows, 3 columns
    fig.suptitle(f"Metadataset: {metadataset_name}", fontsize=base_font_size + 4, y=0.92)

    # Reduce space between subplots and legend
    plt.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.15)

    # Hide empty subplots
    for ax_idx in range(len(metadataset_results), 12):  # 4 rows x 3 columns = 12 subplots
        row, col = divmod(ax_idx, 3)
        axs[row, col].set_visible(False)

    handles, labels = None, None
    for ax_idx, (dataset_name, dataset_results) in enumerate(metadataset_results.items()):
        row, col = divmod(ax_idx, 3)  # 3 columns
        ax = axs[row, col]

        for key in dataset_results.keys():
            all_iterations = []
            all_incumbents = []

            for iteration_values, incumbent_values in dataset_results[key]:
                all_iterations.append(iteration_values)  # Now we append the list instead of extending it
                all_incumbents.append(incumbent_values)  # Same here

            # Convert to a numpy array and average over the first axis (assumed to be over seeds)
            all_incumbents = np.array(all_incumbents)
            mean_incumbents = np.mean(all_incumbents, axis=0)

            # Calculate standard deviation and standard error
            std_dev_incumbents = np.std(all_incumbents, axis=0)
            num_seeds = all_incumbents.shape[0]
            std_err_incumbents = std_dev_incumbents / np.sqrt(num_seeds)

            # Make sure all lists have the same length before plotting
            unique_iterations = sorted(set(all_iterations[0]))  # Assuming all lists of iterations are identical

            # if len(unique_iterations) != len(mean_incumbents):
            #     raise ValueError(f"Mismatched dimensions: {len(unique_iterations)} vs {len(mean_incumbents)}")

            line, = ax.plot(unique_iterations, mean_incumbents, label=str(key))
            ax.fill_between(unique_iterations,
                            mean_incumbents - std_err_incumbents,
                            mean_incumbents + std_err_incumbents,
                            alpha=0.3)

            ax.set_xlim(0, until_iteration)

            # Set the y-axis to log scale
            ax.set_yscale('log')

        if col == 0:
            ax.set_ylabel('Incumbent', fontsize=base_font_size)

        if row == 1:
            ax.set_xlabel('Iteration', fontsize=base_font_size)

        if ax_idx == 0:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_title(f"Dataset: {dataset_name}", fontsize=base_font_size)

    # Common legend below the entire figure
    # Create a new axes for the legend at the bottom of the figure
    legend_ax = fig.add_axes([0.1, 0.45, 0.8, 0.05])  # [left, bottom, width, height]

    # Add the legend to the new axes
    legend_ax.legend(handles, labels, loc='center', ncol=4, fontsize=base_font_size)

    # Hide the axes
    legend_ax.axis('off')

    # Save to disk
    file_path = os.path.join(output_folder, f"{metadataset_name}.png")
    plt.savefig(file_path, bbox_inches='tight')
    print(f"Plot saved to {file_path}")

    # Close the figure to free up memory
    plt.close(fig)