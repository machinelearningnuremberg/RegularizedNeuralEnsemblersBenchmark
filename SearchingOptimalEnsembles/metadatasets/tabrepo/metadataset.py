import numpy as np
import pandas as pd
import torch

from ..base_metadataset import BaseMetaDataset
from ..evaluator import Evaluator
from tabrepo import load_repository, get_context, list_contexts, EvaluationRepository

DATA_VERSION_TO_CONTEXT = {"version0": "D244_F3_C1530_3",
                           "version1": "D244_F3_C1530_10",
                           "version2": "D244_F3_C1530_30",
                           "version3": "D244_F3_C1530_100",
                           "version4": "D244_F3_C1530_175",
                           "version5": "D244_F3_C1530_200",
                           "version6": "D244_F3_C1530"
                                 }

DATA_VERSION_TO_TASK_TYPE = {"class" : "classification",
                             "reg" : "regression"}

TASK_TYPE_TO_METADATASET =  {"classification" : "Supervised Classification",
                             "regression" : "Supervised Regression"}

class TabRepoMetaDataset(Evaluator):
    metadataset_name = "tabrepo"
    def __init__(
        self,
        data_dir: str = None,
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "error",
        context_name: str = "D244_F3_C1530_100",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        #task_type: str = "Supervised Classification", #Supervised Regression or #supervised classification
        data_version: str = "version3_class",
        **kwargs
    ):
        
        self.data_dir = data_dir
        self.seed = seed
        self.split = split
        self.metric_name = metric_name
        self.context_name = DATA_VERSION_TO_CONTEXT[data_version.split("_")[0]]
        self.task_type = DATA_VERSION_TO_TASK_TYPE[data_version.split("_")[1]]
        self.context = get_context(name=context_name)
        self.hp_candidates_dict = self.context.load_configs_hyperparameters()
        self.repo = load_repository(context_name, cache=True)
        self.config_names = np.array(self.get_config_names())
        self.hp_candidates, self.hp_candidates_ids = self._get_hp_candidates_and_indices()

        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
            data_version=data_version
        )
        if self.task_type == "regression":
            #self.metric_name  = "absolute_relative_error"
            self.metric_name = "mse"
        
        self.metadataset_task_type = TASK_TYPE_TO_METADATASET[self.task_type]
        self._initialize()

    def _filter_datasets(self, dataset_names):
        dataset_names = [x for x in dataset_names 
                              if self.repo.dataset_metadata(x)["task_type"]==self.metadataset_task_type]
        return dataset_names
    
    def set_state(self, dataset_name: str, 
                split: str = "valid",
                fold: int = 0):
        self.split = split
        self.fold = fold
        self.dataset_metadata = self.repo.dataset_metadata(dataset_name)
        super().set_state(dataset_name=dataset_name,
                          split=split or self.split)
        self.probing_data = self.get_predictions([[0]])
        self.num_samples = self.probing_data.shape[-2]
        self.num_classes = self.probing_data.shape[-1]

    def get_dataset_names(self) -> list[str]:
        return self._filter_datasets(self.repo.datasets())

    def get_config_names(self) -> list[str]:
        return self.repo.configs()

    def impute_candidates(self, df: pd.DataFrame) -> pd.DataFrame:

        df["layers"] = df.apply(lambda x: str(x.layers), axis=1)

        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number, np.bool_]).columns
        categorical_cols = df.select_dtypes(include=[object]).columns


        # Impute numerical columns with mean
        df[numerical_cols] = df[numerical_cols].apply(lambda col: col.fillna(col.mean()))

        # Impute categorical columns with mode
        df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna("nan").astype(str))

        return df
    
    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        hp_candidates = []
        for pipeline in self.config_names:
            renamed_pipeline = pipeline.replace("_BAG_L1", "")
            hp_candidates.append(
                self.hp_candidates_dict[renamed_pipeline]["hyperparameters"]
            )
        hp_candidates = pd.DataFrame(hp_candidates).drop(columns=["ag_args"])
        hp_candidates = self.impute_candidates(hp_candidates)
        hp_candidates = pd.get_dummies(hp_candidates).astype(float).values

        hp_candidates = torch.FloatTensor( hp_candidates)
        hp_candidates_ids = torch.arange(len(hp_candidates))
        return hp_candidates, hp_candidates_ids
    
    def _posprocess_binary_predictions(self, predictions: np.array) -> np.array:

        if self.dataset_metadata["NumberOfClasses"]==2:
            predictions = torch.cat([1-predictions.unsqueeze(-1),
                                     predictions.unsqueeze(-1)], axis=-1)
                                     
        return predictions
    
    def get_targets(self) -> torch.Tensor:

        """
        Returns the target associated to every sample in the active dataset.

        M: number of samplers per ensemble
        Returns:
            target: torch tensor with the target per sample: [M]    
        """

        if self.split == "valid":
            targets = self.repo.labels_val(dataset=self.dataset_name, fold=self.fold)
        else:
            targets = self.repo.labels_test(dataset=self.dataset_name, fold=self.fold)
        targets = torch.tensor(targets)

        if self.task_type == "classification":
            targets = targets.long()
        elif self.task_type == "regression":
            targets = targets.float()
            
        return targets
    
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
    

    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:
        config_names_in_ensemble = self.config_names[ensembles]
        predictions = []

        #if len(ensembles) == 1:
        #    if len(ensembles[0])==1:
        #        config_names_in_ensemble = [[config_names_in_ensemble]]

        for configs in config_names_in_ensemble:
            if self.split == "valid":
                temp_predictions = torch.FloatTensor(self.repo.predict_val_multi(dataset=self.dataset_name, 
                                                               fold=self.fold,
                                                               configs=configs))
                
            else:
                temp_predictions = torch.FloatTensor(self.repo.predict_test_multi(dataset=self.dataset_name, 
                                                fold=self.fold,
                                                configs=configs))
            if self.task_type == "classification":
                temp_predictions = self._posprocess_binary_predictions(temp_predictions)
            elif self.task_type == "regression":
                temp_predictions = temp_predictions.unsqueeze(-1)
            else:
                raise ValueError("Not valid task type.")
            
            predictions.append(temp_predictions.unsqueeze(0))
        
        predictions = torch.cat(predictions, axis=0)
        num_classes = predictions.shape[-1]
        predictions = torch.nan_to_num(predictions, nan=1./num_classes,
                                       posinf=1.,
                                       neginf=0.)
        return predictions

    def get_num_samples(self) -> int:
        return self.num_samples

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_num_pipelines(self) -> int:
        return len(self.config_names)

    def _get_worst_and_best_performance(self) -> tuple[torch.Tensor, torch.Tensor]:
        #TODO: search the actual worst and best values
        
        if self.metric_name == "neg_roc_auc":
            #to speed up initializatio
            self.worst_performance = 1
            self.best_performance = 0
        else:
            unique_pipelines = self.hp_candidates_ids.unsqueeze(0).T.tolist()
            _, metrics, _, _ = self.evaluate_ensembles(unique_pipelines)

            self.worst_performance = metrics.max().item()
            self.best_performance = metrics.min().item()
        return self.worst_performance, self.best_performance
    
    def get_features(self, ensembles: list[list[int]]) -> torch.Tensor:
        #TODO: find better structure for this function
        return torch.tensor(np.array(self.hp_candidates)[ensembles])
    
    def score_ensemble(self, ensemble: list[int]):
        score, _ = self.repo.evaluate_ensemble(datasets =[self.dataset_name],
                                            configs=self.config_names[ensemble],
                                            rank=False,
                                            folds=[self.fold])

        return score