import numpy as np
import torch
import matplotlib.pyplot as plt

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.metadatasets.scikit_learn.metadataset as slmd
try: 
    import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as trmd
except ImportError:
    trmd = None
import SearchingOptimalEnsembles.metadatasets.nasbench201.metadataset as nbmd

from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler
from sklearn.metrics import f1_score, accuracy_score

from matplotlib import rc
import matplotlib.cm as cm

rc('text', usetex=True)
rc('font', family='serif')

def get_weights_and_scores(dataset_name, dropout_rate):
    metadataset.set_state(dataset_name)
    ne = NeuralEnsembler(metadataset=metadataset,
                            ne_add_y=True,
                            ne_use_context=True,
                            learning_rate=0.0001,
                            ne_epochs=1000,
                            ne_reg_term_div=0.0,
                            ne_reg_term_norm=0.,
                            ne_num_layers=3,
                            ne_num_heads=1,
                            ne_context_size=256,
                            ne_hidden_dim=32,
                            ne_use_mask=True,
                            ne_eval_context_size=128,
                            ne_resume_from_checkpoint=False,
                            ne_unique_weights_per_function=False,
                            ne_dropout_rate=dropout_rate,
                            use_wandb=False,
                            metric_name="error",
                            ne_net_mode="model_averaging",
                            ne_net_type="ffn")

    X_obs = [i.item() for i in metadataset.hp_candidates_ids]

    best_ensemble, best_metric = ne.sample(X_obs)
    weights = ne.get_weights(X_obs)
    predictions = metadataset.get_predictions([X_obs])
    y_true = metadataset.get_targets().numpy()
    ensemble_metric = ne.evaluate_on_split(split="test")

    model_metrics = []
    for i in range(predictions.shape[1]):
        y_pred = predictions[0][i].argmax(-1).numpy()
        #f1_scores.append(f1_score(y_true, y_pred, average="weighted"))
        model_metrics.append(1-accuracy_score(y_true, y_pred))
    model_metrics = np.array(model_metrics)
    model_metrics= (model_metrics-model_metrics.min())/(model_metrics.max()-model_metrics.min())
    ensemble_metric= (ensemble_metric.item()-model_metrics.min())/(model_metrics.max()-model_metrics.min())

    weights = weights[0].mean(-1).mean(-1)
    return weights,model_metrics,ensemble_metric


DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions"
md_class = qmd.QuicktuneMetaDataset
metric_name = "error"
data_version = "micro"
fontsize= 60

metadataset = md_class(
    data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
)

dataset_names = metadataset.get_dataset_names()
fig, axs = plt.subplots(1, 4, figsize=(48, 12))
fig2, axs2 = plt.subplots(1, 4, figsize=(48, 12))

dropout_rates = [0, 0.3, 0.6, 0.9]
max_metric = 0
min_metric = 1
data = {}
for i,dropout_rate in enumerate(dropout_rates):
    temp_dict = {}
    for dataset_name in dataset_names:
        weights, model_metrics, ensemble_metric = get_weights_and_scores(dataset_name, dropout_rate)
        temp_dict[dataset_name] = weights, model_metrics, ensemble_metric
        max_metric = max(max_metric, ensemble_metric)
        min_metric = min(min_metric, ensemble_metric)
    data[dropout_rate] = temp_dict.copy()

norm = plt.Normalize(min_metric, max_metric)  # Assuming integer values are between 1 and 5
cmap = cm.viridis
for i,dropout_rate in enumerate(data.keys()):
    for dataset_name in data[dropout_rate].keys():
        weights, model_metrics, ensemble_metric = data[dropout_rate][dataset_name]

        axs[i].scatter(weights,model_metrics, alpha=0.2, color="blue", s=600)
        axs2[i].scatter(weights,np.ones(len(model_metrics))*ensemble_metric, alpha=0.2, color="blue", s=600)

    axs[i].set_xlabel("Mean Weight Per Model", fontsize=fontsize)
    axs2[i].set_xlabel("Mean Weight Per Model", fontsize=fontsize)

    #axs[i].set_xscale("log")
    #axs[i].set_yscale("log")
    #axs[i].set_xlim(1e-6,1.)
    #axs[i].set_ylim(1e-3,1.)
    
    axs[i].set_xlim(0,1)
    axs[i].set_ylim(0,0.5)

    axs2[i].set_xlim(0,1)
    axs2[i].set_ylim(0,0.5)
    #axs2[i].set_xscale("log")
    #axs2[i].set_yscale("log")

    if i == 0:
        axs[i].set_ylabel("Mean Error",  fontsize=fontsize)
        axs2[i].set_ylabel("Mean Error",  fontsize=fontsize)

    axs[i].set_title(f"Dropout {dropout_rate}", fontsize=fontsize)
    axs[i].tick_params(axis='both', labelsize=fontsize*0.8)
    
    axs2[i].set_title(f"Dropout {dropout_rate}", fontsize=fontsize)
    axs2[i].tick_params(axis='both', labelsize=fontsize*0.8)
    
    #sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    #sm.set_array([])  # No actual data, just the color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # No actual data, just the color bar

    # Add the color bar below the plot
    # if i == 3:
    #     fig2.subplots_adjust(right=0.8)
    #     cbar_ax = fig2.add_axes([1.02, 0.15, 0.02, 0.7]) #left, bottom, width, hegith
    #     cbar = plt.colorbar(sm, orientation='vertical',  cax=cbar_ax)
    #     cbar.set_label('Integer Value')
    plt.tight_layout()
    fig.savefig("saved_plots/hypothesis_overfitting.pdf", bbox_inches='tight')
    fig.savefig("saved_plots/hypothesis_overfitting.png", bbox_inches='tight')
    fig2.savefig("saved_plots/hypothesis_overfitting2.pdf", bbox_inches='tight')
    fig2.savefig("saved_plots/hypothesis_overfitting2.png", bbox_inches='tight')