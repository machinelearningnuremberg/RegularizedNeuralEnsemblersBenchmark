import torch
import pandas as pd
import os
from pathlib import Path

import SearchingOptimalEnsembles.metadatasets.nasbench201.metadataset as nasbench
import SearchingOptimalEnsembles.tests.test_metadataset as base_test
from SearchingOptimalEnsembles.metadatasets.base_metadataset import META_SPLITS

if __name__ == "__main__":

    metadataset = nasbench.NASBench201MetaDataset(data_version="micro")
    for dataset_name in metadataset.dataset_names:
        metadataset.set_state(dataset_name)
        predictions = metadataset.get_predictions([[1,2]])
        metadataset.evaluate_ensembles([[1,2,3],
                                        [4,5,6]])

