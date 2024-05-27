# pylint: disable=all
import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.metadatasets.scikit_learn.metadataset as slmd

from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler


def test_posthoc_ensembling():
    pass


if __name__ == "__main__":
    task_id = 5
    metric_name = "error"
    data_version = "micro"

    name = "quicktune"
    # name = "pipelinebench"

    if name == "quicktune":
        DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions"
        md_class = qmd.QuicktuneMetaDataset
    else:
        DATA_DIR = "/work/dlclarge2/janowski-quicktune/pipeline_bench"
        md_class = slmd.ScikitLearnMetaDataset

    metadataset = md_class(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[task_id])
    num_samples = metadataset.get_num_samples()
    num_classes = metadataset.get_num_classes()

    # Test functionality
    try:
        ensembles = [[1, 2]]
        (
            pipeline_hps,
            metric,
            metric_per_pipeline,
            metric_per_pipeline,
        ) = metadataset.evaluate_ensembles(ensembles=ensembles)
        num_ensembles = len(ensembles)
        num_pipelines = len(ensembles[0])
        weights = np.random.uniform(
            size=(num_ensembles, num_pipelines, num_samples, num_classes)
        )
        weights = torch.FloatTensor(weights)
        metadataset.evaluate_ensembles_with_weights(ensembles=ensembles, weights=weights)

        print("Test passed")
    except Exception as e:
        print(e)

    # get the best performing pipeline
    ensembles = [[i] for i in range(len(metadataset.hp_candidates_ids))]
    pipeline_hps, metric, metric_per_pipeline, _ = metadataset.evaluate_ensembles(
        ensembles=ensembles
    )
    oracle_val_id = metric_per_pipeline.argmin().item()
    oracle_val = metric_per_pipeline.min().item()

    metadataset.set_state(dataset_names[task_id])
    ne = NeuralEnsembler(metadataset=metadataset)

    # X refers to the pipelines
    X_obs = [i for i in range(len(metadataset.hp_candidates_ids))]
    best_ensemble, best_metric = ne.sample(X_obs)

    weights = ne.get_weights(X_obs)
    (
        _,
        best_metric_val,
        metric_per_pipeline_val,
        _,
    ) = metadataset.evaluate_ensembles_with_weights([best_ensemble], weights)

    metadataset_test = md_class(
        data_dir=DATA_DIR,
        metric_name=metric_name,
        data_version=data_version,
        split="test",
    )

    metadataset_test.set_state(dataset_names[task_id])

    ensembles = [[i] for i in range(len(metadataset.hp_candidates_ids))]
    pipeline_hps, metric, metric_per_pipeline, _ = metadataset_test.evaluate_ensembles(
        ensembles=ensembles
    )
    print("Oracle pipeline val:", oracle_val, oracle_val_id)
    print("Oracle pipeline test:", metric_per_pipeline.min().item(), metric_per_pipeline.argmin().item())
    _, val, _, _ = metadataset_test.evaluate_ensembles(ensembles=[[oracle_val_id]])
    print("Oracle from val evaluated on test:", val.item())

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
        metric_per_pipeline_val.min().item(),
        metric_per_pipeline_val.argmin().item(),
    )
    print("Best metric val Neural Ensembler:", best_metric_val.item())

    print(
        "Best metric test using val, single pipeline:",
        metric_per_pipeline_test[0, metric_per_pipeline_val.argmin()].item(),
    )
    print("Best metric test Neural Ensembler:", best_metric_test.item())
