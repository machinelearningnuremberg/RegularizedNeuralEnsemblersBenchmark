# pylint: disable=all
import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd

from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler


def test_posthoc_ensembling():
    pass


if __name__ == "__main__":
    DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions"
    task_id = 5
    metric_name = "error"
    data_version = "micro"
    metadataset = qmd.QuicktuneMetaDataset(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )

    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[task_id])
    ensembles = [[1, 2]]
    num_pipelines, num_samples, num_classes = metadataset.predictions.shape
    num_ensembles = len(ensembles)
    num_pipelines = len(ensembles[0])
    weights = np.random.uniform(
        size=(num_ensembles, num_pipelines, num_samples, num_classes)
    )
    weights = torch.FloatTensor(weights)
    metadataset.evaluate_ensembles_with_weights(ensembles=ensembles, weights=weights)

    ne = NeuralEnsembler(metadataset=metadataset)

    # X refers to the pipelines
    X_obs = np.random.randint(0, 100, 50)
    X_obs = np.arange(120)
    best_ensemble, best_metric = ne.sample(X_obs)

    metadataset.set_state(dataset_names[task_id])
    weights = ne.get_weights(X_obs)
    (
        _,
        best_metric_val,
        metric_per_pipeline_val,
        _,
    ) = metadataset.evaluate_ensembles_with_weights([best_ensemble], weights)

    metadataset_test = qmd.QuicktuneMetaDataset(
        data_dir=DATA_DIR,
        metric_name=metric_name,
        split="test",
        data_version=data_version,
    )
    metadataset_test.set_state(dataset_names[task_id])
    ne.set_state(metadataset=metadataset_test)
    weights = ne.get_weights(X_obs)
    (
        _,
        best_metric_test,
        metric_per_pipeline_test,
        _,
    ) = metadataset_test.evaluate_ensembles_with_weights([best_ensemble], weights)

    print(
        "Best metric val using val, single pipeline:",
        metric_per_pipeline_val.min(),
        metric_per_pipeline_val.argmin(),
    )
    print("Best metric val Neural Ensembler:", best_metric_val)

    print(
        "Best metric test using val, single pipeline:",
        metric_per_pipeline_test[0, metric_per_pipeline_val.argmin()],
    )
    print("Best metric test Neural Ensembler:", best_metric_test)
