import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .custom_metadataset import CustomMetaDataset

class OpenMLMetaDataset(CustomMetaDataset):

    def __init__(
        self,
        data_dir: str = None,
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "error",
        data_version: str = None, #DatasetName
        task_type: str = "Supervised Classification"
    ):
        super().__init__(
            data_dir=data_dir,
            seed=seed,
            split=split,
            metric_name=metric_name,
            data_version=data_version,
            task_type=task_type
        )

    def get_data(self, dataset_name):
        task_id, fold = dataset_name.split("_")
        task_id, fold = int(task_id), int(fold)
        if self.split=="valid":
            (X_train, X_val,
            y_train, y_val) = self.load_train_and_val(task_id, fold)

        elif self.split=="test":
             (self.X_train, self.X_val,
            y_train, y_val) = self.load_train_and_test(task_id, fold)
           
        else:
            raise NameError("Not valid split name.")
        
        return X_train, X_val, y_train, y_val
    
    def set_state(self, dataset_name: str,
                  base_pipelines: list = None):
        X_train, X_val, y_train, y_val = self.get_data(dataset_name)
        super().set_state(
            X_val=X_val,
            y_val=y_val,
            X_train=X_train,
            y_train=y_train,
            base_pipelines=base_pipelines
        )

    def load_train_and_test(self, task_id, fold=0):
        task = openml.tasks.get_task(task_id=task_id)
        assert task.task_type == self.task_type
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
    
        return X_train, X_test, y_train, y_test


    def load_train_and_val(self, task_id, fold=0):
        task = openml.tasks.get_task(task_id=task_id)
        assert task.task_type == self.task_type

        train_indices, _ = task.get_train_test_split_indices(
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
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.33, random_state=self.seed)

        return X_train, X_val, y_train, y_val
