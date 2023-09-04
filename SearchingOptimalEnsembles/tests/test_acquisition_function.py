# pylint: disable=all
# mypy: ignore-errors

import os

import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.samplers.random_sampler as rs
import SearchingOptimalEnsembles.searchers.bayesian_optimization.acquisition as acqf

from ..searchers.bayesian_optimization.models.dre import DRE


def test_acquisition_function(acqf):
    pass


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir, "test_logs"), exist_ok=True)

    checkpoint_path = os.path.join(current_dir, "test_logs")
    device = "cuda"
    data_dir = "/home/pineda/AutoFinetune/aft_data/predictions/"

    metadataset = qmd.QuicktuneMetaDataset(data_dir=data_dir)
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])
    sampler = rs.RandomSampler(metadataset=metadataset, device=torch.device(device))

    model = DRE(
        sampler=sampler,
        checkpoint_path=checkpoint_path,
        device=torch.device(device),
        # dim_in=dim_in,
    )

    model.observed_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
