import os
from pathlib import Path

import pandas as pd
import torch

import RegularizedNeuralEnsemblers.tests.test_metadataset as base_test

from ..metadatasets.base_metadataset import META_SPLITS
from ..metadatasets.custom.custom_metadataset import CustomMetaDataset
from ..metadatasets.custom.openml_metadataset import OpenMLMetaDataset

if __name__ == "__main__":
    metadataset = OpenMLMetaDataset(seed=43)
    metadataset.set_state(dataset_name="31_0")
    metadataset.evaluate_ensembles([[2, 1], [3, 4]])
    print("Done")
