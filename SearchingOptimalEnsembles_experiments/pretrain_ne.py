# pylint: disable=all
import numpy as np

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.metadatasets.scikit_learn.metadataset as slmd
import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as trmd

from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler



if __name__ == "__main__":
    task_id = 4
    metric_name = "error"
    data_version = "version3_class"
    pretrain = True
    DATA_DIR = None
    pretrain_learning_rate = 0.000001
    pretrain_epochs = 1_000_000

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

    run_name = "another_test"
    # X refers to the pipelines
    X_obs = [i for i in range(len(metadataset.hp_candidates_ids))]
    #X_obs = np.random.choice(X_obs, 128)
    X_obs = None

    ne = NeuralEnsembler(metadataset=metadataset,
                         ne_add_y=False,
                         ne_use_context=False,
                         epochs=0,
                         ne_reg_term_div=0,
                         ne_reg_term_norm=0.,
                         ne_num_layers=2,
                         ne_num_heads=1,
                         ne_context_size=32,
                         use_wandb=True,
                         ne_mode="pretraining",
                         ne_hidden_dim=16,
                         ne_checkpoint_name=f"{run_name}.pt",
                         ne_use_mask=False,
                         ne_unique_weights_per_function=True,
                         checkpoint_freq=100,
                         run_name=run_name)

    ne.pretrain_net(X_obs, pretrain_epochs=pretrain_epochs,
                    pretrain_learning_rate=pretrain_learning_rate)