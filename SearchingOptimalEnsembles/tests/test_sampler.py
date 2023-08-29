# pylint: disable=all
import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.samplers.random_sampler as rs
import SearchingOptimalEnsembles.samplers.local_search_sampler as lss


def test_sampler(sampler):
    x = sampler.sample(max_num_pipelines=10, batch_size=16)
    i, j, k = x[0].shape
    assert i == 16
    assert j <= 10

    observed_pipeline_ids = [i * 2 for i in range(20)]
    x = sampler.sample(
        max_num_pipelines=10, batch_size=16, observed_pipeline_ids=observed_pipeline_ids
    )

    # check if all elements in x are in observed_pipeline_ids
    x = sampler.sample(max_num_pipelines=10, batch_size=16)
    i, j, k = x[0].shape
    assert i == 16
    assert j <= 10

    observed_pipeline_ids = [i * 2 for i in range(20)]
    x = sampler.sample(
        max_num_pipelines=10, batch_size=16, observed_pipeline_ids=observed_pipeline_ids
    )

    # check if all elements in x are in observed_pipeline_ids
    x = sampler.sample(max_num_pipelines=10, batch_size=16)
    i, j, k = x[0].shape
    assert i == 16
    assert j <= 10

    observed_pipeline_ids = [i * 2 for i in range(20)]
    x = sampler.sample(
        max_num_pipelines=10, batch_size=16, observed_pipeline_ids=observed_pipeline_ids
    )

    # check if all elements in x are in observed_pipeline_ids
    i, j, k = x[0].shape
    assert i == 16
    assert j <= 10


if __name__ == "__main__":
    DATA_DIR = "/home/pineda/AutoFinetune/aft_data/predictions/"
    metadataset = qmd.QuicktuneMetaDataset(data_dir=DATA_DIR)
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])

    #test random sampler
    sampler = rs.RandomSampler(metadataset=metadataset, device=torch.device("cpu"))
    test_sampler(sampler)

    #test local search sampler
    sampler = lss.LocalSearchSampler(metadataset=metadataset, device=torch.device("cpu"))
    test_sampler(sampler)