# pylint: disable=all
import matplotlib.pyplot as plt
import numpy as np
import torch

import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd

from SearchingOptimalEnsembles.posthoc.des_ensembler import DESEnsembler
from SearchingOptimalEnsembles.metadatasets.custom.openml_metadataset import OpenMLMetaDataset


if __name__ == "__main__":
    metadataset = OpenMLMetaDataset(seed=43)
    metadataset.set_state(dataset_name="31_0")
    ensembler = DESEnsembler(metadataset)
    X_obs = np.arange(metadataset.get_num_pipelines()).tolist()
    ensembler.sample(X_obs)
    score = ensembler.evaluate_on_split()
    print(score)