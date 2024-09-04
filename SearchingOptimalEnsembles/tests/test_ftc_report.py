from pathlib import Path
import sys
import seaborn
import pandas as pd
import numpy as np

from SearchingOptimalEnsembles.metadatasets.ftc.metadataset import FTCMetaDataset
from SearchingOptimalEnsembles.metadatasets.ftc.hub import MODELS
from SearchingOptimalEnsembles.posthoc.single_best import SingleBest


data_dir = "data" 
data_version = "mini"
#data_version = "extended"
metadataset = FTCMetaDataset( metric_name="error",
                             data_version=data_version)
dataset_names = metadataset.get_dataset_names()
dataset_name = dataset_names[0]
results_per_dataset_val = {}
results_per_dataset_test = {}
posthoc_ensembler = SingleBest(metadataset=metadataset)
for dataset_name in dataset_names:
    metadataset.set_state(dataset_name=dataset_name)

    hps = metadataset.row_hp_candidates[dataset_name]
    ids = np.arange(len(hps))
    hps_df = pd.DataFrame(hps)
    results_per_dataset_val[dataset_name] = {}
    results_per_dataset_test[dataset_name] = {}
    incumbent_ensemble, incumbent = posthoc_ensembler.sample(
        ids.tolist()
    )
    for model in MODELS:
        X_obs = ids[hps_df.model ==model]

        if len(X_obs)>0:
            _, metric, metric_per_pipeline, _ = metadataset.evaluate_ensembles([X_obs])
            metric_per_pipeline = metric_per_pipeline.numpy()
            results_per_dataset_val[dataset_name][model]= {
                                                "best_val_model_id": np.argmin(metric_per_pipeline[0]),
                                                "val_metric": np.min(metric_per_pipeline[0])
                                                    }
        else:
            results_per_dataset_val[dataset_name][model]= {
                                                "best_val_model_id": np.nan,
                                                "val_metric": np.nan
                                                    }           
                                            
        
    
    metadataset.set_state(dataset_name=dataset_name,
                          split="test")
    
    for model in MODELS:
        X_obs = ids[hps_df.model ==model]

        if len(X_obs)>0:
            best_val_model_id = X_obs[results_per_dataset_val[dataset_name][model]["best_val_model_id"]]
            _, metric, metric_per_pipeline, _ = metadataset.evaluate_ensembles([[best_val_model_id]])
            metric_per_pipeline = metric_per_pipeline.numpy()
            results_per_dataset_test[dataset_name][model] =metric_per_pipeline[0][0]
        else:
            results_per_dataset_test[dataset_name][model] = np.nan
    
df_results = pd.DataFrame(results_per_dataset_test)
df_results.to_latex(f"results/{data_version}_model_comparion.tex")
print("Done.")


    