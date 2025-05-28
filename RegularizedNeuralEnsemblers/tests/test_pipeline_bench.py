import numpy as np
import torch

import RegularizedNeuralEnsemblers.metadatasets.scikit_learn.metadataset as skmd

from ..posthoc.des_ensembler import DESEnsembler

if __name__ == "__main__":
    metadataset = skmd.ScikitLearnMetaDataset()
    datasets = metadataset.get_dataset_names()
    metadataset.set_state(datasets[0], split="train")

    # ensembler = DESEnsembler(metadataset)
    # X_obs = np.arange(metadataset.get_num_pipelines()).tolist()
    # ensembler.sample(X_obs)
    # test_metric = ensembler.evaluate_on_split(split="test")
    # print(test_metric)
