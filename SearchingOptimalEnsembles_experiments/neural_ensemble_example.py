# SearchingOptimalEnsembles_experiments/neural_ensemble_example.py
# Demonstrates how to train a neural ensemble on TabRepo metadataset.

import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as trmd
from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler

if __name__ == "__main__":
    data_version = "version3_class"
    metric_name = "nll"
    task_id = 0  # or any valid index

    metadataset = trmd.TabRepoMetaDataset(
        data_dir=None, metric_name=metric_name, data_version=data_version
    )

    dataset_names = metadataset.meta_splits["meta-test"]  # or whichever split
    metadataset.set_state(dataset_names[task_id])

    # Initialize the neural ensembler
    neural_ensembler = NeuralEnsembler(
        metadataset=metadataset,
    )

    # Candidate pipelines (in practice, you'd sample or load these)
    X_obs = [[1], [2], [3], [4], [5], [6], [7], [8]]

    # Now sample an ensemble with learned dynamic weights
    best_ensemble, best_metric = neural_ensembler.sample(X_obs)
    weights = neural_ensembler.get_weights(X_obs)

    _, metric_val, _, _ = metadataset.evaluate_ensembles_with_weights(
        ensembles=[best_ensemble], weights=weights
    )
    print("Neural ensemble:", best_ensemble)
    print("Neural ensemble metric:", metric_val.item())
