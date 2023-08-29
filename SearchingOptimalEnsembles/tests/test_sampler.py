# pylint: disable=all


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
