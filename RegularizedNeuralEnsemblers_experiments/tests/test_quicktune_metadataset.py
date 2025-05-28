# import torch
import SearchingOptimalEnsembles.metadatasets.quicktune_metadataset as qmd
from base_test import test_evaluate_ensembles, test_get_batch


def test_quicktune_metadataset():
    metadataset = qmd.QuickTuneMetaDataset()
    test_evaluate_ensembles(metadataset)
    test_get_batch(metadataset)
