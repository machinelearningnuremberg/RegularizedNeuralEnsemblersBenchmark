import matplotlib.pyplot as plt
import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.posthoc.sk_stacker as sks
import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as tabrepod

DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions/"

if __name__ == "__main__":
    metadataset = qmd.QuicktuneMetaDataset(data_dir=DATA_DIR)

    metadataset = tabrepod.TabRepoMetaDataset(#data_version="version3_class")
                                              data_version="version3_reg")
    metadataset.set_state(dataset_name=metadataset.get_dataset_names()[0])
    X_obs = [1,2,3,4,5,6]

    model_names = ["random_forest", "gradient_boosting", "linear_model", "svm"]
    
    for model_name in model_names:
        stacker = sks.ScikitLearnStacker(metadataset=metadataset,
                                         model_name=model_name)
        ensemble, val_metric = stacker.sample(X_obs)
        test_metric=stacker.evaluate_on_split(split="test")
        print(val_metric, test_metric)
    print("Done.")