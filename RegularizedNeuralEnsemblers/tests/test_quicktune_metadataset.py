# pylint: disable=all

import torch

import RegularizedNeuralEnsemblers.metadatasets.quicktune.metadataset as qmd
import RegularizedNeuralEnsemblers.tests.test_metadataset as base_test

# DATA_DIR = "/home/pineda/AutoFinetune/aft_data/predictions/"
DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions/"


def test_quicktune_metadataset():
    metadataset = qmd.QuicktuneMetaDataset(data_dir=DATA_DIR)
    base_test.test_evaluate_ensembles(metadataset)


if __name__ == "__main__":
    test_quicktune_metadataset()
