# pylint: disable=all

import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.samplers.random_sampler as rs
import SearchingOptimalEnsembles.tests.test_sampler as base_test

if __name__ == "__main__":
    DATA_DIR = "/home/pineda/AutoFinetune/aft_data/predictions/"
    metadataset = qmd.QuicktuneMetaDataset(data_dir=DATA_DIR)
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])
    sampler = rs.RandomSampler(metadataset=metadataset, device=torch.device("cpu"))
    base_test.test_sampler(sampler)
