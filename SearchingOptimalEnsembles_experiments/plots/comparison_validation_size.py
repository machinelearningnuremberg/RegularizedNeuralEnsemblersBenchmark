import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "SearchingOptimalEnsembles_experiments/reports/report37/"
processed_table = pd.read_csv(path+"processed_table.csv")
processed_table.set_index('experiment_name', inplace=True)

experiment_ids = [
    "RQ6_ftc_extended_nll" ,
    "RQ6_qt_mini_nll",
    "RQ6_qt_micro_nll",
    "RQ6_nb_mini_nll",
    "RQ6_nb_micro_nll",
    "RQ6_tr_version3class_nll",
    "RQ6_tr_version3reg_mse"
    ]
experiment_id = "RQ6_tr_version3class_nll"
methods = ["neural", "greedy"]

dict_names= {
    "RQ6_ftc_extended_nll" : "FTC",
    "RQ6_qt_mini_nll" : "QT-Mini",
    "RQ6_qt_micro_nll": "QT-Micro",
    "RQ6_nb_mini_nll": "NB (100)",
    "RQ6_nb_micro_nll": "NB (1000)",
    "RQ6_tr_version3class_nll": "TR-Class",
    "RQ6_tr_version3reg_mse": "TR-Reg"
}


dict_names_methods = {
    "neural" : "NE-Stack",
    "greedy" : "Greedy"
}

fig, axes = plt.subplots(4,2, figsize=(5, 10))  # Adjust figsize as needed

for k, experiment_id in enumerate(experiment_ids):

    j = k//2
    i = k%2
    ax = axes[j,i]

    selected_columns = []
    for column in processed_table.columns:
        if experiment_id in column:
            selected_columns.append(column)
    results = {}
    for method in methods:
        results[method] = []

    for value in processed_table.index:
        for method in methods:
            if method in value:
                results[method].append(np.nanmean(processed_table.loc[value,selected_columns].values) )

    for method in methods:
        ax.plot(results[method], label= dict_names_methods[method])
    ax.set_title(dict_names[experiment_id])
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels([1,5,10,25,50,100])

    ax.set_xlabel("Pct. Valid. Data")

    if i == 0:
        ax.set_ylabel("Normalized NLL")

handles, labels = axes[0, 0].get_legend_handles_labels()

# Add a common legend
fig.legend(handles, labels, loc='lower center', ncol=2,  bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig("SearchingOptimalEnsembles_experiments/plots/saved_plots/comparison_validation_size.png",  bbox_inches='tight')
plt.savefig("SearchingOptimalEnsembles_experiments/plots/saved_plots/comparison_validation_size.pdf",  bbox_inches='tight')

print("Done.")
