# pylint: disable=all
import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.metadatasets.scikit_learn.metadataset as slmd
try: 
    import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as trmd
except ImportError:
    trmd = None
import SearchingOptimalEnsembles.metadatasets.nasbench201.metadataset as nbmd

from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler


def test_posthoc_ensembling():
    pass


if __name__ == "__main__":
    metric_name = "error"
    data_version = "micro"
    pretrain = False
    DATA_DIR = None
    pretrain_epochs = 100_000
    checkpoint_name = "auto"
    checkpoint_name = None

    # name = "quicktune"
    # name = "tabrepo"
    # name = "pipelinebench"
    name = "nasbench201"

    if name == "quicktune":
        DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions"
        md_class = qmd.QuicktuneMetaDataset
    elif name == "tabrepo":
        md_class = trmd.TabRepoMetaDataset
    elif name == "pipelinebench":
        DATA_DIR = "/work/dlclarge2/janowski-quicktune/pipeline_bench"
        md_class = slmd.ScikitLearnMetaDataset
    elif name == "nasbench201":
        DATA_DIR = "/work/dlclarge2/janowski-quicktune/nb201_data"
        md_class = nbmd.NASBench201MetaDataset
    else:
        raise ValueError("Unknown metadataset")

    metadataset = md_class(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )
    
    if name == "nasbench201":
        task_id = 0
        dataset_names = metadataset.get_dataset_names()
    else:
        task_id = 8
        dataset_names = metadataset.meta_splits["meta-test"]
    metadataset.set_state(dataset_names[task_id])
    num_samples = metadataset.get_num_samples()
    num_classes = metadataset.get_num_classes()

    # Test functionality
    try:
        # choose random ids from metadataset.hp_candidates_ids
        ensembles = [
            np.random.choice(metadataset.hp_candidates_ids, size=1, replace=False).tolist()
            for _ in range(5)  
        ]
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
    ensembles = [[i.item()] for i in metadataset.hp_candidates_ids]
    pipeline_hps, metric, metric_per_pipeline, _ = metadataset.evaluate_ensembles(
        ensembles=ensembles
    )
    oracle_val_id = metric_per_pipeline.argmin().item()
    oracle_val = metric_per_pipeline.min().item()

    # X refers to the pipelines
    X_obs = [i.item() for i in metadataset.hp_candidates_ids]

    ne = NeuralEnsembler(metadataset=metadataset,
                         ne_add_y=True,
                         ne_use_context=True,
                         epochs=1000,
                         ne_reg_term_div=0.,
                         ne_reg_term_norm=0.,
                         ne_num_layers=4,
                         ne_num_heads=4,
                         ne_context_size=24,
                         ne_eval_context_size=1,
                         ne_checkpoint_name=checkpoint_name,
                         use_wandb=False)
    

    if pretrain:
        ne.pretrain_net(X_obs, pretrain_epochs=pretrain_epochs)
        
    metadataset.set_state(dataset_names[task_id])

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

    ensembles = [[i.item()] for i in metadataset.hp_candidates_ids]
    pipeline_hps, metric, metric_per_pipeline, _ = metadataset_test.evaluate_ensembles(
        ensembles=ensembles
    )
    print("Oracle pipeline val:", oracle_val, oracle_val_id)
    print("Oracle pipeline test:", metric_per_pipeline.min().item(), metric_per_pipeline.argmin().item())
    _, val, _, _ = metadataset_test.evaluate_ensembles(ensembles=[[oracle_val_id]])
    print("Oracle from val evaluated on test:", val.item())

    ne.set_state(metadataset=metadataset_test)

    X_context, y_context = ne.get_context(X_obs, metadataset=metadataset)

    weights = ne.get_weights(X_obs, X_context=X_context, y_context=y_context)
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
