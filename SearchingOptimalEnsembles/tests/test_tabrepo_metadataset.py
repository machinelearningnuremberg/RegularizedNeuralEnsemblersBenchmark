import torch

import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as tabrepod
import SearchingOptimalEnsembles.tests.test_metadataset as base_test


if __name__ == "__main__":

    metadataset = tabrepod.TabRepoMetaDataset()
    metadataset.set_state(metadataset.dataset_names[0])
    metadataset.evaluate_ensembles([[1,2,3],
                                    [4,5,6]])

    metadataset.score_ensemble([1,2])
    base_test.test_evaluate_ensembles(metadataset)