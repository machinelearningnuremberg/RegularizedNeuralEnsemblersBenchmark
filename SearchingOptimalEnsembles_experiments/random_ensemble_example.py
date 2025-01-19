# SearchingOptimalEnsembles_experiments/random_ensemble_example.py
# Demonstrates how to build a random ensemble on TabRepo metadataset.

import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as trmd
from SearchingOptimalEnsembles.posthoc.random_ensembler import RandomEnsembler

if __name__ == "__main__":
    data_version = "version3_class"
    metric_name = "nll"
    task_id = 0  # or any valid index

    metadataset = trmd.TabRepoMetaDataset(
        data_dir=None, metric_name=metric_name, data_version=data_version
    )
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])

    # Initialize random sampler
    ensembler = RandomEnsembler(metadataset=metadataset, device=torch.device("cpu"))

    # Candidate pipelines (in practice, you'd sample or load these)
    X_obs = [[1], [2], [3], [4], [5], [6], [7], [8]]

    best_ensemble, best_metric = ensembler.sample(X_obs)
    print("Best random ensembler found:", best_ensemble)
    print("Random ensembler metric:", best_metric)
