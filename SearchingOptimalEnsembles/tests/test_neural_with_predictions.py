import torch
import pandas as pd
import os
from pathlib import Path

from SearchingOptimalEnsembles.metadatasets.custom.custom_metadataset import CustomMetaDataset
from SearchingOptimalEnsembles.metadatasets.custom.openml_metadataset import OpenMLMetaDataset
import SearchingOptimalEnsembles.tests.test_metadataset as base_test
from SearchingOptimalEnsembles.metadatasets.base_metadataset import META_SPLITS
from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler

if __name__ == "__main__":

    metadataset = OpenMLMetaDataset(seed=43)
    datasets = metadataset.get_dataset_names()
    metadataset.set_state(dataset_name=datasets[0])
    predictions = metadataset.get_predictions([[1,2]])
    targets = metadataset.get_targets()

    X = []
    for i in range(predictions.shape[1]):
        X.append(predictions[0, i].numpy())
    y = targets.numpy()

    ne = NeuralEnsembler()
    ne.fit(X, y)
    ne.predict(X)
    print("Done")
