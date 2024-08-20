from __future__ import annotations

import torch
from SearchingOptimalEnsembles.metadatasets.custom.openml_metadataset import OpenMLMetaDataset
from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler

if __name__ == "__main__":
    mode = "combined"  # "combined_conditional"
    metadataset = OpenMLMetaDataset(seed=43)
    datasets = metadataset.get_dataset_names()
    metadataset.set_state(dataset_name=datasets[0])
    predictions = metadataset.get_predictions([[1, 2]])
    targets = metadataset.get_targets()

    # X is a list of numpy arrays with predictions from different models
    # y is the ground truth

    X = []
    for i in range(predictions.shape[1]):
        X.append(predictions[0, i].numpy())
    y = targets.numpy()

    ne = NeuralEnsembler(mode=mode, device=torch.device("cpu"))
    ne.fit(X, y)
    ne.predict(X)
    ne.predict_proba(X) # TODO: Needed!
    print("Done")
