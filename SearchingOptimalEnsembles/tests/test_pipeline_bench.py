
import torch
import numpy as np

import SearchingOptimalEnsembles.metadatasets.scikit_learn.metadataset as skmd
from SearchingOptimalEnsembles.posthoc.des_ensembler import DESEnsembler

if __name__ == "__main__":

    metadataset = skmd.ScikitLearnMetaDataset()
    datasets = metadataset.get_dataset_names()
    metadataset.set_state(datasets[0])
    pipelines = metadataset.get_pipelines([[1,2],[2,3]])
    ensembler = DESEnsembler(metadataset)
    X_obs = np.arange(metadataset.get_num_pipelines()).tolist()
    ensembler.sample(X_obs)
