# SearchingOptimalEnsembles_experiments/greedy_ensemble_example.py
# Demonstrates how to build a greedy ensemble on TabRepo metadataset.

import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as trmd
from SearchingOptimalEnsembles.posthoc.greedy_ensembler import GreedyEnsembler

if __name__ == "__main__":
    data_version = "version3_class"
    metric_name = "nll"
    task_id = 0  # or any valid index

    metadataset = trmd.TabRepoMetaDataset(
        data_dir=None, metric_name=metric_name, data_version=data_version
    )
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])

    # Initialize greedy ensembler
    ensembler = GreedyEnsembler(metadataset=metadataset, device=torch.device("cpu"))

    # Candidate pipelines (in practice, you'd sample or load these)
    X_obs = [[1], [2], [3], [4], [5], [6], [7], [8]]

    best_ensemble, best_metric = ensembler.sample(X_obs)
    print("Greedy ensemble found:", best_ensemble)
    print("Greedy ensemble metric:", best_metric)
