import numpy as np
import pandas as pd
import torch

from ..base_metadataset import BaseMetaDataset
from ..evaluator import Evaluator
from .pipeline_samplers.random import RandomPipelineSampler

class CustomMetaDataset(Evaluator):
    """This MetaDataset contains the evaluations of a specific dataset.
    The evaluations are built after the creation of the object given a set of base pipelines.
    If the base pipelines are not given, they are randomly initialized
    """
    
    metadataset_name = "custom"

    def __init__(
        self,
        data_dir: str = None,
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "error",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        data_version: str = None, #DatasetName
        task_type: str = "classification",
        device: str = 'cpu',
        num_base_pipelines: int = 20,
        **kwargs
    ):
        self.data_dir = data_dir
        self.seed = seed
        self.split = split
        self.metric_name = metric_name
        self.data_version = data_version
        self.task_type = task_type
        self.num_base_pipelines = num_base_pipelines
        self.base_pipelines = None
        self.device = device
        self.dataset_name = None

        if self.task_type == "regression":
            self.metric_name  = "mse"
          
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
            device=device,
            data_version=data_version
        )

    def fit_base_pipelines(self):
        for base_pipeline in self.base_pipelines:
            print(base_pipeline)
            base_pipeline.fit(self.X_train, self.y_train)

    def get_features(self, ensembles):
        return self.hp_candidates[ensembles]

    def set_state(self,
                  X_val: np.array,
                  y_val: np.array,
                  X_test: np.array = None,
                  y_test: np.array = None,
                  X_train: np.array = None,
                  y_train: np.array = None,
                  base_pipelines: list = None,
                  hp_candidates: np.array = None,
                  ):
        """
        If base_pipelines are passed, they should be already fit.
        If X_train, y_train are passed, the base pipelines will be refit.
        If not base_pipelines are passed they will be generated randomly and trained on X_train
        """
        self.X_val = X_val
        self.y_val = y_val
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if base_pipelines is None:
            #random pipelines
            if self.base_pipelines is None:
                self.base_pipelines, self.hp_candidates = RandomPipelineSampler(self.num_base_pipelines,
                                                                        random_state=self.seed,
                                                                        task_type=self.task_type).sample()

        else:
            self.base_pipelines = base_pipelines
            
            if hp_candidates is not None:
                self.hp_candidates = hp_candidates

        if X_train is not None and y_train is not None:
            self.fit_base_pipelines()

        self.num_pipelines = len(self.base_pipelines)
        self.num_classes = len(np.unique(y_val))
        self.num_samples = len(y_val)
        self.hp_candidates_ids = torch.FloatTensor(np.arange(self.num_pipelines))
        self.hp_candidates = torch.FloatTensor(self.hp_candidates)
        self.dataset_names = []

        if self.task_type == "classification":
            self.targets = torch.LongTensor(self.y_val)
        else:
            self.targets = torch.FloatTensor(self.y_val)

        self.precompute_predictions()


    def get_base_pipelines(self):
        return self.base_pipelines

    def get_dataset_names(self) -> list[str]:
        return self.dataset_names

    def precompute_predictions(self):

        predictions = []
        for pipeline in self.base_pipelines:

            if self.task_type == "classification":
                predictions.append(
                    torch.FloatTensor(
                        pipeline.predict_proba(self.X_val)
                    ).unsqueeze(0)
                )
            elif self.task_type == "regression":
                 predictions.append(
                    torch.FloatTensor(
                        pipeline.predict(self.X_val)
                    ).unsqueeze(0)
                )               
            else:
                raise NameError("Not valid task type.")
            
        self.predictions = torch.cat(predictions)

        if self.task_type == "classification":
            self.predictions = torch.nan_to_num(self.predictions, nan=1./self.num_classes,
                                        posinf=1.,
                                        neginf=0.)
        
    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hp_candidates, self.hp_indices
    
    def get_targets(self) -> torch.Tensor:
        return self.targets

    def get_num_pipelines(self) -> int:
        return self.num_pipelines
    
    def get_num_classes(self) -> int:
        return self.num_classes

    def _get_worst_and_best_performance(self) -> tuple[torch.Tensor, torch.Tensor]:
        unique_pipelines = self.hp_candidates_ids.unsqueeze(0).T.tolist()
        _, metrics, _, _ = self.evaluate_ensembles(unique_pipelines)

        self.worst_performance = metrics.max().item()
        self.best_performance = metrics.min().item()
        return self.worst_performance, self.best_performance

    def get_time(self, ensembles: list[list[int]]) -> torch.Tensor:
        """
        Returns the target associated to every sample in the active dataset.

        B: number of ensembles
        N: number of models per ensemble

        Args:
            ensembles: List of list with the base model index to evaluate: [B, N]

        Returns:
            time: torch tensor with the time per pipeline and ensemble: [B, N]    
        """
    
        return torch.zeros(len(ensembles), 
                            len(ensembles[0]))
    
    def get_predictions(self, ensembles: list):
        return self.predictions[torch.LongTensor(ensembles)]
    
    def get_X_and_y_val(self):
        return self.X_val, self.y_val
    
    def get_X_and_y_test(self):
        return self.X_test, self.y_test

    def get_num_samples(self):
        if self.split == "valid":
            return len(self.X_val)
        elif self.split == "test":
            return len(self.X_test)
        else:
            raise ValueError("Not valid split.")
    

    def score(self, y_pred, y_test):

        if self.metric_name=="error":
            return (y_pred.argmax(-1)!=y_test).mean()   
        elif self.metric_name =="mse":
            return ((y_pred-y_test)**2).mean()   
        else:
            raise NameError("Metric named is not defined in the score funciton.") 