import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .custom_metadataset import CustomMetaDataset

OPENML_TASKS = [11, 15, 18, 22, 23, 29, 31, 37]
OPENML_TASK_TYPE_MAP = {
    "Supervised Classification": "classification",
    "Supervised Regression": "regression"
}
class OpenMLMetaDataset(CustomMetaDataset):

    metadataset_name = "openml"

    def __init__(
        self,
        data_dir: str = None,
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "error",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        data_version: str = None, #DatasetName
        task_type: str = "classification",
        num_base_pipelines: int = 20,
        **kwargs
    ):
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
            data_version=data_version,
            num_base_pipelines=num_base_pipelines,
            task_type=task_type
        )
        self._initialize()
      
    def get_dataset_names(self, num_folds=10) -> list[str]:
        dataset_names = []
        for task in OPENML_TASKS:
            for i in range(num_folds):
                dataset_names.append(f"{task}_{i}")
        return dataset_names
    
    def get_data(self, dataset_name):
        task_id, fold = dataset_name.split("_")
        task_id, fold = int(task_id), int(fold)
        (X_train, X_val, X_test,
        y_train, y_val, y_test
        ) = self.load_splits(task_id, fold)

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_X_and_y (self):
        if self.split == "train":
            return self.X_train, self.y_train
        elif self.split == "valid":
            return self.X_val, self.y_val
        else:
            return self.X_test, self.y_test

    def set_state(self, dataset_name: str,
                  split = "valid",
                  base_pipelines: list = None,
                  hp_candidates: np.array = None):
        self.split = split
        
        if dataset_name != self.dataset_name:
            X_train, X_val, X_test, y_train, y_val, y_test = self.get_data(dataset_name)
            self.dataset_name = dataset_name
            super().set_state(
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                X_train=X_train,
                y_train=y_train,
                base_pipelines=base_pipelines,
                hp_candidates=hp_candidates
            )

    def load_splits(self, task_id, fold=0):
        task = openml.tasks.get_task(task_id=task_id)
        assert OPENML_TASK_TYPE_MAP[task.task_type] == self.task_type
        train_indices, test_indices = task.get_train_test_split_indices(
            repeat=0,
            fold=fold,
            sample=0,
        )
        X, y = task.get_X_and_y(dataset_format="dataframe")

        X =  pd.get_dummies(
                        X
                    ).fillna(0) \
                    .astype(float).values
        y = LabelEncoder().fit_transform(y)
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.33, random_state=self.seed)
    
        return X_train, X_val, X_test, y_train, y_val, y_test