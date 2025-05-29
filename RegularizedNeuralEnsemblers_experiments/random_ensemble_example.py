# RegularizedNeuralEnsemblers_experiments/random_ensemble_example.py
# Demonstrates how to build a random ensemble on a metadataset.

import RegularizedNeuralEnsemblers.metadatasets.quicktune.metadataset as qmd
import torch
from RegularizedNeuralEnsemblers.posthoc.random_ensembler import RandomEnsembler

if __name__ == "__main__":
    data_version = "micro"
    metric_name = "nll"
    task_id = 0  # or any valid index
    #Use your "path/to/quicktune/predictions"
    DATA_DIR =  "/work/dlclarge2/janowski-quicktune/predictions"

    metadataset = qmd.QuicktuneMetaDataset(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])

    # Initialize random sampler
    ensembler = RandomEnsembler(
        metadataset=metadataset,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Candidate pipelines (in practice, you'd sample or load these)
    X_obs = [[1], [2], [3], [4], [5], [6], [7], [8]]

    best_ensemble, best_metric = ensembler.sample(X_obs)
    print("Best random ensemble found:", best_ensemble)
    print("Random ensembler metric:", best_metric)
