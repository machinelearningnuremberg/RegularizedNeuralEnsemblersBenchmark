# pylint: disable=all
# mypy: ignore-errors
import os

import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.samplers.random_sampler as rs
import SearchingOptimalEnsembles.searchers.bayesian_optimization.acquisition as acqf

from ..searchers.bayesian_optimization.models.dre import DRE


def test_acquisition(acqf):
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
    pending_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    acqf = acqf.ExpectedImprovement(device=torch.device(device))
    acqf.set_state(model)

    ensembles = sampler.generate_ensembles(
        candidates=np.array(model.observed_ids), num_pipelines=3, batch_size=16
    )

    new_pipelines = sampler.generate_ensembles(
        candidates=np.array(pending_ids), num_pipelines=1, batch_size=16
    )

    candidates = np.concatenate((model.observed_ids, pending_ids)).tolist()
    pipeline_hps, _, metric_per_pipeline, _ = metadataset.evaluate_ensembles(ensembles)
    new_pipeline_hps, _, _, _ = metadataset.evaluate_ensembles(new_pipelines)
    new_metric_per_pipeline = torch.zeros(len(new_pipeline_hps), 1)

    pipeline_hps = torch.cat((pipeline_hps, new_pipeline_hps), dim=1)
    metric_per_pipeline = torch.cat((metric_per_pipeline, new_metric_per_pipeline), dim=1)

    score = acqf.eval(
        x=pipeline_hps.to(torch.device(device)),
        metric_per_pipeline=metric_per_pipeline.to(torch.device(device)),
    )
