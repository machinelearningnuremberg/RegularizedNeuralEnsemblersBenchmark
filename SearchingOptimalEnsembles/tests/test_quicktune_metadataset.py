import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import torch
import SearchingOptimalEnsembles.tests.base_test as base_test

DATA_DIR = "/home/pineda/AutoFinetune/aft_data/predictions/"

def test_quicktune_metadataset():
    metadataset = qmd.QuicktuneMetaDataset(data_dir = DATA_DIR)
    base_test.test_evaluate_ensembles(metadataset)


if __name__ == "__main__":
    test_quicktune_metadataset()
