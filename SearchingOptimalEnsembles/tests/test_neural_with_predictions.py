from SearchingOptimalEnsembles.metadatasets.custom.custom_metadataset import CustomMetaDataset
from SearchingOptimalEnsembles.metadatasets.custom.openml_metadataset import OpenMLMetaDataset
import SearchingOptimalEnsembles.tests.test_metadataset as base_test
from SearchingOptimalEnsembles.metadatasets.base_metadataset import META_SPLITS
from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler

if __name__ == "__main__":

    mode = "combined" # "combined_conditional"
    metadataset = OpenMLMetaDataset(seed=43)
    datasets = metadataset.get_dataset_names()
    metadataset.set_state(dataset_name=datasets[0])
    predictions = metadataset.get_predictions([[1,2]])
    targets = metadataset.get_targets()

    #X is a list of numpy arrays with predictions from different models
    #y is the ground truth

    X = []
    for i in range(predictions.shape[1]):
        X.append(predictions[0, i].numpy())
    y = targets.numpy()

    ne = NeuralEnsembler(mode=mode)
    ne.fit(X, y)
    ne.predict(X)
    print("Done")
