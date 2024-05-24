import numpy as np
import torch
from typing_extensions import Literal

from ..metadatasets import MetaDatasetMapping
from ..metadatasets.base_metadataset import META_SPLITS, BaseMetaDataset
from ..posthoc.base_ensembler import BaseEnsembler
from ..utils.common import instance_from_map


def eval(
    ensemble: list,
    metadataset_name: Literal["scikit-learn", "nasbench201", "quicktune"],
    dataset_id: int = 0,
    meta_split_id: int = 0,
    data_version: str = "micro",
    metric_name: str = "nll",
    split: str = "test",
    ensembler: BaseEnsembler = None,
    observed_pipelines: np.array = None,
    device: torch.device = torch.device("cuda"),
    val_metadataset: BaseMetaDataset = None,
):
    metadataset_args = {
        "meta_split_ids": META_SPLITS[meta_split_id],
        "metric_name": metric_name,
        "data_version": data_version,
        "split": split,
    }
    metadataset = instance_from_map(
        MetaDatasetMapping,
        metadataset_name,
        name="metadataset",
        kwargs=metadataset_args,
    )

    dataset_name = metadataset.meta_splits["meta-test"][dataset_id]
    metadataset.set_state(dataset_name=dataset_name)
    ensembler.set_state(metadataset=metadataset, device=ensembler.device)

    weights = None
    if ensembler is not None:
        if hasattr(ensembler, "get_weights"):
            # if ensemble is None:
            #    ensemble = np.arange(len(metadataset.hp_candidates_ids)).tolist()
            
            X_context, y_context = ensembler.get_context(ensemble, val_metadataset)
            ensembler.update_context_size(100)
            weights = ensembler.get_weights(
                ensemble, X_context=X_context, y_context=y_context
            )
            print(weights.mean())

    if weights is None:
        _, metric, metric_per_pipeline, _ = metadataset.evaluate_ensembles([ensemble])
    else:
        _, metric, metric_per_pipeline, _ = metadataset.evaluate_ensembles_with_weights(
            [ensemble], weights
        )

    return metric, metric_per_pipeline
