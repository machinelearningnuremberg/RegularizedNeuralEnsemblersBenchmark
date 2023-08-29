def test_evaluate_ensembles(metadataset):
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_dataset_name(dataset_names[0])
    (
        pipeline_hps,
        metric,
        metric_per_pipeline,
        time_per_pipeline,
    ) = metadataset.evaluate_ensembles([[0, 1, 2], [3, 4, 5]])

    assert pipeline_hps.shape[:2] == (2, 3)
    assert metric.shape == (2,)
    assert metric_per_pipeline.shape == (2, 3)
    assert time_per_pipeline.shape == (2, 3)

    (
        pipeline_hps,
        metric,
        metric_per_pipeline,
        time_per_pipeline,
    ) = metadataset.evaluate_ensembles([[0], [1], [2]])

    assert pipeline_hps.shape[:2] == (3, 1)
    assert metric.shape == (3,)
    assert metric_per_pipeline.shape == (3, 1)
    assert time_per_pipeline.shape == (3, 1)

    (
        pipeline_hps,
        metric,
        metric_per_pipeline,
        time_per_pipeline,
    ) = metadataset.evaluate_ensembles([[0]])

    assert pipeline_hps.shape[:2] == (1, 1)
    assert metric.shape == (1,)
    assert metric_per_pipeline.shape == (1, 1)
    assert time_per_pipeline.shape == (1, 1)


# def test_get_batch(metadataset):
#     pass
