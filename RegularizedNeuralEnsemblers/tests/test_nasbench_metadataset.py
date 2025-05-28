import os
from pathlib import Path

import pandas as pd
import torch

import RegularizedNeuralEnsemblers.metadatasets.nasbench201.metadataset as nasbench
import RegularizedNeuralEnsemblers.tests.test_metadataset as base_test

from ..metadatasets.base_metadataset import META_SPLITS

if __name__ == "__main__":
    metadataset = nasbench.NASBench201MetaDataset(data_version="micro")
    for dataset_name in metadataset.dataset_names:
        metadataset.set_state(dataset_name)
        predictions = metadataset.get_predictions([[1, 2]])
        metadataset.evaluate_ensembles([[1, 2, 3], [4, 5, 6]])
