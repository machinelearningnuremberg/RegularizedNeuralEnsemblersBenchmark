# pylint: disable=all
# mypy: ignore-errors

import os

import torch

#from ..metadatasets.quicktune.metadataset import QuicktuneMetaDataset
#from ..samplers.random_sampler import RandomSampler
#from ..searchers.bayesian_optimization.models.dre import DRE
#from ..searchers.bayesian_optimization.searcher import BayesianOptimization as BO

from SearchingOptimalEnsembles.metadatasets.quicktune.metadataset import QuicktuneMetaDataset
from SearchingOptimalEnsembles.samplers.random_sampler import RandomSampler
from SearchingOptimalEnsembles.samplers.local_search_sampler import LocalSearchSampler
from SearchingOptimalEnsembles.searchers.bayesian_optimization.models.dre import DRE
from SearchingOptimalEnsembles.searchers.bayesian_optimization.searcher import BayesianOptimization as BO

if __name__ == "__main__":

    experiment_id = "test7"
    device = "cuda"
    data_dir = "/home/pineda/AutoFinetune/aft_data/predictions/"
    surrogate_name = "dre"
    max_num_pipelines = 4
    surrogate_args = {"criterion_type": "weighted_listwise", "add_y": 1}
    acquisition_args = {"beta": 0.}
    acquisition_name = "ucb"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    worker_dir = os.path.join(current_dir, "test_logs", experiment_id)
    os.makedirs(worker_dir, exist_ok=True)

    metadataset = QuicktuneMetaDataset(data_dir=data_dir)
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])
    sampler = RandomSampler(metadataset=metadataset, device=torch.device(device))

    dim_in = sampler.metadataset.hp_candidates.shape[1]

    bo_searcher = BO(
        metadataset=metadataset,
        surrogate_name=surrogate_name,
        acquisition_name=acquisition_name,
        acquisition_args=acquisition_args,
        surrogate_args=surrogate_args,
        worker_dir=worker_dir,
        checkpoint_path=worker_dir
    )

    #bo_searcher.meta_train_surrogate(num_epochs=1000, valid_frequency=50)
    bo_searcher.run(max_num_pipelines=1, meta_num_epochs = 0, num_inner_epochs=100)

    print("Done!")
