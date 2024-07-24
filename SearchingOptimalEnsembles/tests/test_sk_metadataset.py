import torch
import pandas as pd
import os
from pathlib import Path

import SearchingOptimalEnsembles.metadatasets.scikit_learn.metadataset as skm
import SearchingOptimalEnsembles.tests.test_metadataset as base_test
from SearchingOptimalEnsembles.metadatasets.base_metadataset import META_SPLITS

if __name__ == "__main__":

    metadataset = skm.ScikitLearnMetaDataset(metric_name="error")
    dataset_name = metadataset.get_dataset_names()[0]
    metadataset.set_state(dataset_name)
    predictions = metadataset.get_predictions([[1,2]])
    metadataset.evaluate_ensembles([[1,2,3],
                                    [4,5,6]])

    metadataset.get_pipelines([[1],[2]])
    print("Done.")