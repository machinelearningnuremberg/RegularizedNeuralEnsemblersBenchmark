from __future__ import annotations

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_ensembler import BaseEnsembler

MODELS = {
    "classification" :
    {
        "random_forest" : RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "linear_model" : LogisticRegression,
        "svm": SVC
    },
    "regression": {
        "random_forest" : RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "linear_model" : LinearRegression,
        "svm": SVR      
    }
}
DEFAULT_MODEL_ARGS = {
 "classification": {
      "svm": {"probability":True}
 }
}

class ScikitLearnStacker(BaseEnsembler):
    """ScikitLearn Stacker"""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        sks_model_name: str = "random_forest",
        sks_model_args: dict | None = None,
        device: torch.device = torch.device("cpu"),
        normalize_performance: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)
        self.model_name = sks_model_name
        self.model_args = sks_model_args
        self.normalize_performance = normalize_performance


        if hasattr(metadataset, "task_type"):
            self.task_type = metadataset.task_type
        else:
            self.task_type = "classification"

        if self.model_args is None:
            if self.task_type in DEFAULT_MODEL_ARGS.keys():
                self.model_args = DEFAULT_MODEL_ARGS[self.task_type].get(self.model_name, {})
            else:
                self.model_args = {}
        self.model = MODELS[self.task_type][self.model_name](**self.model_args)
        self.absolute_relative_error = lambda y_true,y_pred: torch.abs(y_true-y_pred)/torch.max(torch.ones(1),torch.abs(y_true))
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def sample(self, X_obs, **kwargs):
        self.X_obs = X_obs
        self.best_ensemble = X_obs

        base_functions = (
            self.metadataset.get_predictions([X_obs])[0]
            .transpose(0, 1)
            .transpose(2, 1)
        ).numpy()
        num_samples, num_classes, num_pipelines = base_functions.shape
        base_functions = base_functions.reshape(-1, num_classes * num_pipelines)
        y_true = self.metadataset.get_targets()
        self.model.fit(base_functions, y_true)

        if self.task_type == "classification":
            y_pred = torch.FloatTensor(self.model.predict_proba(base_functions))
        else:
            y_pred = torch.FloatTensor(self.model.predict(base_functions))
           
        metric = self.metadataset.score_y_pred(y_pred, y_true)

        return self.best_ensemble, metric

    def get_metric(self, base_functions, y_true):
        
        y_pred = torch.FloatTensor(self.model.predict_proba(base_functions))
        y_true = torch.tensor(y_true)
        if self.metadataset.metric_name == "nll":
            metric = self.cross_entropy(y_pred, y_true)
        elif self.metadataset.metric_name == "error":
            metric = (y_pred.argmax(-1) != y_true).float().mean()
        elif self.metadataset.metric_name == "absolute_relative_error":
            metric = self.absolute_relative_error(y_true, y_pred)
        return metric

    def evaluate_on_split(self, split: str = "test"):

        self.metadataset.set_state(dataset_name=self.metadataset.dataset_name,
                                    split = split)
        base_functions = (
            self.metadataset.get_predictions([self.best_ensemble])[0]
            .transpose(0, 1)
            .transpose(2, 1)
        ).numpy()
        num_samples, num_classes, num_pipelines = base_functions.shape
        base_functions = base_functions.reshape(-1, num_classes * num_pipelines)
        y_true = self.metadataset.get_targets() 

        if self.task_type == "classification":
            y_pred = torch.FloatTensor(self.model.predict_proba(base_functions))
        else:
            y_pred = torch.FloatTensor(self.model.predict(base_functions))
           
        metric = self.metadataset.score_y_pred(y_pred, y_true)

        if self.normalize_performance:
            metric = self.metadataset.normalize_performance(metric)

        return metric