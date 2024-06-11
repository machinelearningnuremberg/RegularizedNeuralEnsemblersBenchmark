from __future__ import annotations

import torch

from ..evaluator import Evaluator

import pandas as pd

import numpy as np
import torch

from pathlib import Path

from tqdm import tqdm

from torchvision import datasets as dset
from torchvision import transforms
import torch
from pathlib import Path
from tqdm import tqdm

# TODO: confirm with Arber
def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return transform

class NASBench201MetaDataset(Evaluator):

    _preds = pd.DataFrame()
    _dataset = pd.DataFrame()
    _configs = pd.DataFrame()

    data_version_map = {
        "micro": 200,
        "mini": 1000,
        "extended": 15625
    }


    num_classes = {}
    splits = ["val", "test"]

    def __init__(
        self,
        data_dir: str = "/work/dlclarge2/janowski-quicktune/nb201_data",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 0,
        split: str = "valid",
        metric_name: str = "error",
        data_version: str = "micro",
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
        )

        self.feature_dim = 6
        self.data_dir = Path(data_dir)
        self.data_version = data_version

        self._initialize()

    def set_state(self, dataset_name: str):
        self.logger.debug(f"Setting dataset: {dataset_name}")

        assert dataset_name in self.get_dataset_names(), f"Invalid dataset name: {dataset_name}"

        transform = _data_transforms_cifar10()
        self.load_dataset(transform)
        self.load_data(dataset_name)

        super().set_state(dataset_name)

    # TODO: we should save dataframes to disk and load them from there
    def load_dataset(self, transform):
        # TODO: ask Arber how he generated the data

        nb201_seeds = [777, 888, 999]
        torch.manual_seed(nb201_seeds[self.seed])
        # Load the full training dataset (intended for train) but use for valid/test split
        full_dataset = dset.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)

        # Splitting the dataset
        num_samples = len(full_dataset)
        valid_size = int(0.5 * num_samples)
        test_size = int(0.2 * num_samples)
        _, valid_test_dataset = torch.utils.data.random_split(full_dataset, [num_samples - valid_size - test_size, valid_size + test_size])
        valid_dataset, test_dataset = torch.utils.data.random_split(valid_test_dataset, [valid_size, test_size])

        dataset = valid_dataset if self.split == "valid" else test_dataset
        self._dataset = self.create_dataframe(dataset, self.split)

    def create_dataframe(self, dataset, split):
        data = []
        for image, label in tqdm(dataset, desc=f"Loading {split} data"):
            img_array = np.array(image.permute(1, 2, 0))  # Convert CxHxW to HxWxC
            flat_img = img_array.flatten()
            data.append({
                'data': flat_img,
                'label': label,
                'split': split
            })
        return pd.DataFrame(data)

    # TODO: we should save dataframes to disk and load them from there
    def load_data(self, dataset):
        
        split = "val" if self.split == "valid" else self.split

        data_path = self.data_dir / "nb201_preds" / dataset / str(self.seed)
        # Retrieve all model files
        model_files = list(data_path.glob("*.pt"))

        # Randomize model files order
        np.random.shuffle(model_files)

        # Use only up to the number of models specified in data_version_map
        selected_model_files = model_files[:self.data_version_map[self.data_version]]

        all_data = []
        all_configs = []
        new_model_id = 1  # Start indexing from 1

        for model_file in tqdm(selected_model_files, desc=f"Processing {dataset}-{self.seed}-{split}"):
            data = torch.load(model_file)
            # Use a new sequential model_id instead of file stem
            preds = data['preds'][split].tensors[0].numpy()

            # Create an index with new_model_id and an index range for each datapoint
            index = pd.MultiIndex.from_product([[new_model_id], range(len(preds))], names=['model_id', 'datapoint_id'])
            df = pd.DataFrame(preds, index=index, columns=[f'pred_class_{i}' for i in range(preds.shape[1])])
            all_data.append(df)

            config = self.parse_config(data['config'])
            index = pd.Index([new_model_id], name='model_id')
            config_df = pd.DataFrame([config], index=index)
            all_configs.append(config_df)

            new_model_id += 1  # Increment model_id for the next model

        # Concatenate all dataframes ensuring they all have the same structure
        if all_data:
            self._preds = pd.concat(all_data)

        if all_configs:
            configs_df = pd.concat(all_configs)
            self._configs = self.one_hot_encode_configs(configs_df)

    def parse_config(self, config_str):
        # Assume the configuration is split into meaningful parts here
        ops = [op for op in config_str.split('|') if op and op != '+']
        return {f'op{i}': op for i, op in enumerate(ops) if op}

    def get_dataset_names(self) -> list[str]:
        return ["cifar10"]

    def one_hot_encode_configs(self, configs_df):
        # Assuming configuration keys are the column headers
        all_ops = pd.get_dummies(configs_df, prefix='', prefix_sep='').groupby(level=0).max()
        return all_ops

    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._configs.empty:
            raise ValueError("Configuration data is empty. Ensure data is loaded correctly.")

        # Convert numpy arrays to torch tensors
        hp_candidates = self._configs.values
        hp_candidates = torch.tensor(hp_candidates, dtype=torch.float32)

        hp_candidates_ids = list(self._configs.index)
        hp_candidates_ids = torch.tensor(hp_candidates_ids, dtype=torch.int32)

        return hp_candidates, hp_candidates_ids

    def _get_worst_and_best_performance(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._preds.empty:
            raise ValueError("Prediction data is empty. Ensure data is loaded and processed correctly.")

        # Dictionary to store accuracy of each model
        accuracy_per_model = {}

        for model_id, group in self._preds.groupby(level='model_id'):
            # Determine predicted labels from predictions for each model
            prediction_cols = [col for col in group.columns if col.startswith('pred_class_')]
            predicted_labels = group[prediction_cols].idxmax(axis=1)
            predicted_labels = predicted_labels.str.replace('pred_class_', '').astype(int)

            # Filtering the labels from _dataset based on the current split and datapoint_id
            # relevant_dataset = self._dataset[(self._dataset['split'] == self.split) & (self._dataset.index.isin(group.index.get_level_values('datapoint_id')))]
            relevant_dataset = self._dataset[(self._dataset.index.isin(group.index.get_level_values('datapoint_id')))]
            true_labels = torch.tensor(relevant_dataset['label'].values)

            # Compute correct predictions and calculate accuracy
            correct_predictions = predicted_labels.values == true_labels.values
            accuracy = correct_predictions.astype(float).mean()  # Convert Boolean array to float and calculate mean

            accuracy_per_model[model_id] = accuracy

        # Convert performance to a tensor for easier manipulation
        performance_tensor = torch.tensor(list(accuracy_per_model.values()))

        # Get worst and best performance
        worst_performance = performance_tensor.min()
        best_performance = performance_tensor.max()

        return worst_performance, best_performance

    def get_num_classes(self) -> int:
        return 10

    def get_num_samples(self) -> int:
        return len(self._preds.index.get_level_values('datapoint_id').unique())

    def get_features(self, ensembles: list[list[int]]) -> torch.Tensor:

        # Flatten the list of lists to get all model IDs in ensembles
        model_ids = [model_id for ensemble in ensembles for model_id in ensemble]
        # Use `.loc` to select rows from _configs based on model_ids, ensuring all model_ids are present in _configs
        features_df = self._configs.loc[model_ids]
        # Convert the selected DataFrame to a torch Tensor
        features_tensor = torch.tensor(features_df.values, dtype=torch.float32)

        return features_tensor

    def get_targets(self) -> torch.Tensor:
        return torch.tensor(self._dataset['label'].values, dtype=torch.int32)

    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:

        y_proba_np = self._get_probabilities(ensembles=ensembles)
        # Convert the numpy array to torch tensor
        y_proba = torch.tensor(y_proba_np, dtype=torch.float32)
        # Assuming the current shape of y_proba is (B, M, N, C)
        y_proba = y_proba.permute(0, 2, 1, 3)

        return y_proba

    # TODO: efficiency!
    def _get_probabilities(self, ensembles: list[list[int]]) -> np.ndarray:

        num_datapoints = self.get_num_samples()
        num_classes = self.get_num_classes()

        # Adjust the shape to include ensemble size, even if each ensemble has 1 model at a time
        y_proba = np.zeros((len(ensembles), num_datapoints, len(ensembles[0]), num_classes))

        for i, ensemble in enumerate(ensembles):
            for j, model_id in enumerate(ensemble):
                # Selecting predictions for a specific model across all its datapoints
                model_preds = self._preds.loc[model_id]
                # Ensure predictions are aligned correctly by datapoint_id
                for index, (_, preds) in enumerate(model_preds.iterrows()):
                    # Assuming index matches datapoint_id if they are 0 to num_datapoints-1 consecutively
                    y_proba[i, index, j, :] = preds.values  # Note the addition of 'j' to index ensemble members

        return y_proba

    def get_time(self, ensembles: list[list[int]]) -> torch.Tensor:
        return torch.zeros(len(ensembles), len(ensembles[0]))
