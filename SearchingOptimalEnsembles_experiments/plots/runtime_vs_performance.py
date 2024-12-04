import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')

translate_dict = {
    "single1_0": "Single-Best",
    "random1_0": "Random",
    "topm1_0": "Top5",
    "topm1_1": "Top50",
    "quick1_1": "Quick",
    "greedy1_1": "Greedy",
    "cmaes1_0": "CMAES",
    "sks1_0": "Random Forest",
    "sks1_1": "Gradient Boosting",
    "sks1_3": "SVM",
    "sks1_2": "Linear",
    "neural4_0": "MA",
    "divbo1_1": "DivBO",
    "leo1_1": "EO",
    "neural3_4": "NE-Stack",
    "neural3_5": "NE-MA"
}

fontsize= 15
path = "SearchingOptimalEnsembles_experiments/reports/report25/"
processed_table = pd.read_csv(path+"processed_table.csv")
runtimes = pd.read_csv(path+"runtimes.csv")
processed_table.set_index('experiment_name', inplace=True)
runtimes.set_index('experiment_name', inplace=True)

processed_table = processed_table.median(axis=1)

merged_df = pd.merge(runtimes, processed_table.to_frame(), left_index=True, right_index=True)
merged_df = merged_df.drop(["divbo1_1"])
merged_df = merged_df.drop(["leo1_1"])
merged_df = merged_df.drop(["random1_0"])

x = merged_df["mean"] #runtime
y = merged_df[0] #performance

plt.figure()
plt.xscale("log")
plt.yscale("log")

plt.scatter(x, y)

labels = merged_df.index
texts = []
for i, label in enumerate(labels):
    texts.append(
        plt.text(x[i], y[i], translate_dict[label],  ha='center', fontsize=fontsize)
    )
adjust_text(texts)
plt.xlabel("Avg. Runtime", fontsize=fontsize)
plt.ylabel("Avg. Normalized NLL", fontsize=fontsize)
plt.grid(True, which="both", ls="-", color='0.65',zorder=1)
plt.savefig("SearchingOptimalEnsembles_experiments/plots/saved_plots/runtime_vs_performance.png",  bbox_inches='tight')
plt.savefig("SearchingOptimalEnsembles_experiments/plots/saved_plots/runtime_vs_performance.pdf",  bbox_inches='tight')
