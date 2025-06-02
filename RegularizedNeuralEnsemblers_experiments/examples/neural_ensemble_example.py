# RegularizedNeuralEnsemblers_experiments/neural_ensemble_example.py
# Demonstrates how to train a neural ensemble on a metadataset.

import torch

import RegularizedNeuralEnsemblers.metadatasets.quicktune.metadataset as qmd
from RegularizedNeuralEnsemblers.posthoc.neural_ensembler import NeuralEnsembler

if __name__ == "__main__":
    data_version = "micro"
    metric_name = "nll"
    task_id = 0  # or any valid index
    # Use your "path/to/quicktune/predictions"
    DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions"

    metadataset = qmd.QuicktuneMetaDataset(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )

    dataset_names = metadataset.meta_splits["meta-test"]
    metadataset.set_state(dataset_names[task_id])

    # Initialize the neural ensembler
    neural_ensembler = NeuralEnsembler(
        metadataset=metadataset,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Candidate pipelines (in practice, you'd sample or load these)
    X_obs = [1, 2, 3, 4, 5]

    # Now fits an ensemble with learned dynamic weights
    best_ensemble, best_metric = neural_ensembler.sample(X_obs)
    weights = neural_ensembler.get_weights(X_obs)

    _, metric_val, _, _ = metadataset.evaluate_ensembles_with_weights(
        ensembles=[best_ensemble], weights=weights
    )
    print("Best ensemble found by Neural Ensembler:", best_ensemble)
    print("Neural ensemble metric:", metric_val.item())
