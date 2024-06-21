import torch
import pandas as pd
import os
from pathlib import Path

from SearchingOptimalEnsembles.metadatasets.custom.custom_metadataset import CustomMetaDataset
from SearchingOptimalEnsembles.metadatasets.custom.openml_metadataset import OpenMLMetaDataset
import SearchingOptimalEnsembles.tests.test_metadataset as base_test
from SearchingOptimalEnsembles.metadatasets.base_metadataset import META_SPLITS

if __name__ == "__main__":

    metadataset = OpenMLMetaDataset(seed=43)
    metadataset.set_state(dataset_name="31_0")
    metadataset.evaluate_ensembles([[2,1],[3,4]])
    print("Done")