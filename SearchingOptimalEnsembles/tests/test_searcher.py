# pylint: disable=all
# mypy: ignore-errors

import os

import torch

from ..metadatasets.quicktune.metadataset import QuicktuneMetaDataset
from ..samplers.random_sampler import RandomSampler
from ..searchers.bayesian_optimization.models.dre import DRE
from ..searchers.bayesian_optimization.searcher import BayesianOptimization as BO

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir, "test_logs"), exist_ok=True)

    checkpoint_path = os.path.join(current_dir, "test_logs")
    device = "cuda"
    data_dir = "/home/pineda/AutoFinetune/aft_data/predictions/"

    metadataset = QuicktuneMetaDataset(data_dir=data_dir)
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])
    sampler = RandomSampler(metadataset=metadataset, device=torch.device(device))

    dim_in = sampler.metadataset.hp_candidates.shape[1]
    surrogate_name = "dre"
    surrogate_args = {"dim_in": dim_in}

    bo_searcher = BO(
        metadataset=metadataset,
        checkpoint_path=checkpoint_path,
        surrogate_name=surrogate_name,
        surrogate_args=surrogate_args,
    )

    bo_searcher.meta_train_surrogate(num_epochs=10000, valid_frequency=50)

    print("Done!")
