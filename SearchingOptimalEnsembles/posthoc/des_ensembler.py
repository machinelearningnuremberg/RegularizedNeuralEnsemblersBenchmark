from __future__ import annotations

import numpy as np
import torch
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES
from deslib.des.knop import KNOP

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_ensembler import BaseEnsembler

METHOD_TO_CLASS = {"KNORAE": KNORAE,
                   "MetaDES": METADES,
                   "KNOP": KNOP}


class DESEnsembler(BaseEnsembler):

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
        des_method_name: str = "KNORAE",
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)
        
        assert hasattr(self.metadataset, "get_base_pipelines") \
                and hasattr(self.metadataset, "get_X_and_y"),\
                      "Metadataset does not have the required attributes."
        
        self.des_method_name = des_method_name

    def sample(
        self,
        X_obs,
        # max_num_pipelines: int = 5,
        # num_batches: int = 5,
        # num_suggestions_per_batch: int = 1000,
        **kwargs,
    ) -> tuple[list, float]:
        
        self.X_obs = X_obs
        base_pipelines = self.metadataset.get_base_pipelines()

        selected_pipelines = [base_pipelines[x] for x in X_obs]
        self.des_object = METHOD_TO_CLASS[self.des_method_name](selected_pipelines)

        self.metadataset.set_state(dataset_name=self.metadataset.dataset_name,
                                   split="valid")       
        X_dsel, y_dsel = self.metadataset.get_X_and_y()
        self.des_object.fit(X_dsel, y_dsel)
        y_pred = self.des_object.predict_proba(X_dsel)
        
        if self.metadataset.task_type == "regression":
            y_dsel = torch.FloatTensor(y_dsel)
        else:
            y_dsel = torch.LongTensor(y_dsel)
        y_pred = torch.FloatTensor(y_pred)       
        
        best_metric = self.metadataset.score_y_pred(y_pred, y_dsel)
        self.best_ensemble = X_obs

        return X_obs, best_metric

    def evaluate_on_split(self, split: str ="test") -> float:
        #TODO CHange only state
        self.metadataset.set_state(dataset_name=self.metadataset.dataset_name,
                                   split=split)
        X_dsel, y_dsel = self.metadataset.get_X_and_y()
        y_pred = self.des_object.predict_proba(X_dsel)
        if self.metadataset.task_type == "regression":
            y_dsel = torch.FloatTensor(y_dsel)
        else:
            y_dsel = torch.LongTensor(y_dsel)
        y_pred = torch.FloatTensor(y_pred)
        best_metric = self.metadataset.score_y_pred(y_pred, y_dsel)
        
        return best_metric

       

 