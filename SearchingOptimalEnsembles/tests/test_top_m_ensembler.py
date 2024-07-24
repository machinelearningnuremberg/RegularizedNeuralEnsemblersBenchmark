import matplotlib.pyplot as plt
import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd
import SearchingOptimalEnsembles.posthoc.top_m_ensembler as tme
import SearchingOptimalEnsembles.posthoc.quick_greedy_ensembler as qge

DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions/"

if __name__ == "__main__":
    metadataset = qmd.QuicktuneMetaDataset(data_dir=DATA_DIR)
    metadataset.set_state(dataset_name=metadataset.get_dataset_names()[0])
    X_obs = [1,2,3,4,5,6]
    ensembler = qge.QuickGreedyEnsembler(metadataset)
    ensemble, best_metric = ensembler.sample(X_obs)
