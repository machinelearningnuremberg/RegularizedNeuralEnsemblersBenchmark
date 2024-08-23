import torch
import pandas as pd
import os
from pathlib import Path

import SearchingOptimalEnsembles.metadatasets.tabrepo.metadataset as tabrepod
import SearchingOptimalEnsembles.tests.test_metadataset as base_test
from SearchingOptimalEnsembles.metadatasets.base_metadataset import META_SPLITS

if __name__ == "__main__":

    metadataset = tabrepod.TabRepoMetaDataset(#data_version="version3_class")
                                              data_version="version3_class",
                                              metric_name="nll")
    metadataset.set_state(metadataset.dataset_names[0])
    metadataset.evaluate_ensembles([[1,2,3],
                                    [4,5,6]])

    metadataset.score_ensemble([1,2])
    metadataset._get_worst_and_best_performance()

    current_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    dataset_sizes = []

    for i, meta_split_ids in META_SPLITS.items():
        metadataset = tabrepod.TabRepoMetaDataset(meta_split_ids=meta_split_ids)
        
        for j, dataset_name in enumerate(metadataset.meta_splits["meta-test"]):
            metadataset.set_state(dataset_name)
            print(dataset_name)
            dataset_sizes.append((f"{j}-{i}", dataset_name, metadataset.get_num_samples(), metadataset.get_num_classes()))
    
    pd.DataFrame(dataset_sizes).to_csv(current_file_path / ".." / ".." / "SearchingOptimalEnsembles_experiments" 
                                       / "plots" / "saved_results" / "dataset_sizes.csv")

    base_test.test_evaluate_ensembles(metadataset)