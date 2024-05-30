# pylint: disable=all
import numpy as np

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.metadatasets.scikit_learn.metadataset as slmd
import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as trmd

from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler



if __name__ == "__main__":
    task_id = 5
    metric_name = "error"
    data_version = "micro"
    pretrain = True
    DATA_DIR = None
    pretrain_learning_rate = 0.00001
    pretrain_epochs = 800_000

    name = "tabrepo"

    if name == "quicktune":
        DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions"
        md_class = qmd.QuicktuneMetaDataset
    elif name == "tabrepo":
        md_class = trmd.TabRepoMetaDataset
    else:
        DATA_DIR = "/work/dlclarge2/janowski-quicktune/pipeline_bench"
        md_class = slmd.ScikitLearnMetaDataset

    metadataset = md_class(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )


    # X refers to the pipelines
    X_obs = [i for i in range(len(metadataset.hp_candidates_ids))]

    ne = NeuralEnsembler(metadataset=metadataset,
                         ne_add_y=True,
                         ne_use_context=True,
                         epochs=0,
                         ne_reg_term_div=0.,
                         ne_reg_term_norm=0.,
                         ne_num_layers=4,
                         ne_num_heads=4,
                         ne_context_size=128,
                         use_wandb=True)

    if pretrain:
        ne.pretrain_net(X_obs, pretrain_epochs=pretrain_epochs,
                        pretrain_learning_rate=pretrain_learning_rate)
        