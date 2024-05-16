# pylint: disable=all
import os

import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.samplers.random_sampler as rs

from ..searchers.bayesian_optimization.models.dre import DRE


def test_DRE():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir, "test_logs"), exist_ok=True)

    checkpoint_path = os.path.join(current_dir, "test_logs")
    device = "cuda"
    data_dir = "/home/pineda/AutoFinetune/aft_data/predictions/"
    data_dir = "/work/dlclarge2/janowski-quicktune/predictions"

    metadataset = qmd.QuicktuneMetaDataset(data_dir=data_dir)
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])
    sampler = rs.RandomSampler(metadataset=metadataset, device=torch.device(device))
    # dim_in = sampler.metadataset.hp_candidates.shape[1]

    model = DRE(
        sampler=sampler,
        checkpoint_path=checkpoint_path,
        device=torch.device(device),
        # dim_in=dim_in,
    )

    mean_parameters_before = list(model.parameters())[0].mean().item()
    loss = model.fit(10)
    mean_parameters_after = list(model.parameters())[0].mean().item()

    assert mean_parameters_before != mean_parameters_after
    assert loss != 0.0
    assert torch.isnan(loss) == False


if __name__ == "__main__":
    test_DRE()
