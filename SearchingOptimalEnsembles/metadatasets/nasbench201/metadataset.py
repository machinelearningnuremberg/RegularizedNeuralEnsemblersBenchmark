from __future__ import annotations

from pathlib import Path

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import random
import torch
from scipy.special import softmax
from tqdm import tqdm

from ..evaluator import Evaluator


class NASBench201MetaDataset(Evaluator):
    _predictions = pd.DataFrame()
    _labels = pd.DataFrame()
    _features = pd.DataFrame()

    data_version_map = {"micro": 100, "mini": 1000, "extended": 15625}

    num_classes = {}
    splits = ["val", "test"]

    def __init__(
        self,
        data_dir: str = "/work/dlclarge2/janowski-quicktune/nb201_data/nb201_preds",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 0,
        split: str = "valid",
        metric_name: str = "error",
        data_version: str = "micro",
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
        )

        self.feature_dim = None
        self.data_dir = Path(data_dir)
        self.data_version = data_version

        self._initialize()

    def set_state(self, dataset_name: str, split: str ="valid"):
        self.logger.debug(f"Setting dataset: {dataset_name}")

        assert (
            dataset_name in self.get_dataset_names()
        ), f"Invalid dataset name: {dataset_name}"

        self.load_data(dataset_name, split)

        super().set_state(dataset_name, split)

    def load_data(self, dataset, split):
        split = "val" if split == "valid" else split
        base_path = self.data_dir / dataset / str(self.seed) / self.data_version

        labels_path = base_path / f"labels_{split}.parquet"
        features_path = base_path / f"features_{split}.parquet"
        preds_path = base_path / f"preds_{split}.parquet"

        # if labels_path.exists() and features_path.exists() and preds_path.exists():
        #     self._labels = dd.read_parquet(labels_path)
        #     self._features = dd.read_parquet(features_path)
        #     self._predictions = dd.read_parquet(preds_path)
        #     self.feature_dim = len(self._features.columns)
        #     return

        base_path.mkdir(parents=True, exist_ok=True)

        data_path = self.data_dir / dataset / str(self.seed)
        model_files = list(data_path.glob("*.pt"))[
            : self.data_version_map[self.data_version]
        ]
        random.shuffle(model_files)

        all_preds = []
        all_labels = None
        all_configs = []

        model_id = 0
        for model_file in tqdm(
            model_files, desc=f"Processing {dataset}-{self.seed}-{split}"
        ):
            data = torch.load(model_file)

            logits = data["preds"][split].tensors[0].numpy()
            # Applying softmax to logits to obtain probabilities
            preds = softmax(logits, axis=1)
            df_preds = pd.DataFrame(
                preds, columns=[f"pred_class_{i}" for i in range(preds.shape[1])]
            )
            df_preds["model_id"] = model_id
            df_preds["datapoint_id"] = range(
                len(preds)
            )  # Consistent ordering of datapoints
            all_preds.append(df_preds)

            if all_labels is None:
                labels = data["preds"][split].tensors[1].numpy()
                all_labels = pd.DataFrame(labels, columns=["label"])
                all_labels["datapoint_id"] = range(len(labels))

            config = self.parse_config(data["config"])
            df_configs = pd.DataFrame([config])
            df_configs["model_id"] = model_id
            all_configs.append(df_configs)

            model_id += 1

        df_preds_combined = pd.concat(all_preds)
        df_configs_combined = pd.concat(all_configs)
        df_configs_combined = self.one_hot_encode_operations(df_configs_combined)

        self._predictions = dd.from_pandas(df_preds_combined, npartitions=10)
        self._features = dd.from_pandas(df_configs_combined, npartitions=10)
        self.feature_dim = len(self._features.columns)

        self._labels = dd.from_pandas(all_labels, npartitions=1)

        try:
            self._predictions.to_parquet(preds_path)
            self._features.to_parquet(features_path)
            self._labels.to_parquet(labels_path)
        except:
            pass

    def parse_config(self, config_str):
        # Assume the configuration is split into meaningful parts here
        ops = [op for op in config_str.split("|") if op and op != "+"]
        return {f"op{i}": op for i, op in enumerate(ops) if op}

    def get_dataset_names(self) -> list[str]:
        return ["cifar10", "cifar100", "imagenet"]

    def one_hot_encode_operations(self, configs_df):
        model_ids = configs_df["model_id"]
        # Flatten the series to get a list of all unique operations across all columns
        all_ops = pd.unique(configs_df.values.ravel())

        # Create a DataFrame to hold the one-hot encoded vectors
        encoded_df = pd.DataFrame(index=configs_df.index)

        # Encode each operation column
        for column in configs_df.columns:
            if column == "model_id":
                continue  # Skip model_id as it's not an operation
            # Get one-hot encoding for the column
            one_hot = pd.get_dummies(configs_df[column], dtype=int)  # Ensure dtype is int
            # Add missing columns for operations that are not present in this column
            missing_cols = set(all_ops) - set(one_hot.columns)
            for col in missing_cols:
                one_hot[col] = 0  # These are initialized as int implicitly
            # Prefix the column names to ensure they are unique
            one_hot = one_hot.rename(
                columns={op: f"{column}_{op}" for op in one_hot.columns}
            )
            encoded_df = pd.concat([encoded_df, one_hot], axis=1)

        encoded_df["model_id"] = model_ids
        return encoded_df

    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._features is None or self._features.compute().empty:
            raise ValueError(
                "Configuration data is empty. Ensure data is loaded correctly."
            )
        # Convert the Dask DataFrame to a Dask Array
        # Note: `lengths=True` will ensure that the array knows the length of each block
        hp_candidates_da = self._features.to_dask_array(lengths=True)

        # Computing the values with Dask, and then convert to a Torch tensor
        # Note: Ensure that the compute operation is feasible; it must not exceed your system's memory capacity
        hp_candidates_np = (
            hp_candidates_da.compute()
        )  # This will pull the data into memory as a NumPy array
        hp_candidates = torch.tensor(hp_candidates_np, dtype=torch.float32)

        # Extracting model IDs and converting them similarly
        hp_candidates_ids_np = self._features["model_id"].compute().values
        hp_candidates_ids = torch.tensor(hp_candidates_ids_np, dtype=torch.int32)

        return hp_candidates, hp_candidates_ids

    def _get_worst_and_best_performance(self):
        split = "val" if self.split == "valid" else self.split
        base_path = self.data_dir / self.dataset_name / str(self.seed) / self.data_version
        metafeatures_path = base_path / f"metafeatures_{split}.csv"

        # Check if metafeatures file exists
        # if metafeatures_path.exists():
        #     # Load previously computed performance metrics
        #     metafeatures = pd.read_csv(metafeatures_path)
        #     worst_performance = metafeatures['worst_performance'].item()
        #     best_performance = metafeatures['best_performance'].item()
        # else:
        # Compute performance metrics
        if self._predictions is None or self._predictions.compute().empty:
            raise ValueError(
                "Prediction data is empty. Ensure data is loaded and processed correctly."
            )

        num_classes = self.get_num_classes()
        true_labels = self._labels["label"].compute().values.astype(int)

        def compute_accuracy(group):
            predicted_probs = group.values.reshape(-1, num_classes)
            predicted_labels = predicted_probs.argmax(axis=1)
            correct_predictions = predicted_labels == true_labels
            # Ensure that correct_predictions is an array and perform mean calculation
            if isinstance(correct_predictions, np.ndarray):
                accuracy = np.mean(correct_predictions.astype(float))
            else:
                accuracy = float(
                    correct_predictions
                )  # Handle the case where it's a single bool
            return accuracy

        accuracy_series = self._predictions.groupby("model_id").apply(
            compute_accuracy, meta=("x", "f8")
        )
        accuracies = accuracy_series.compute().tolist()

        # Save performance metrics
        accuracies_tensor = torch.tensor(accuracies)
        worst_performance = accuracies_tensor.min().item()
        best_performance = accuracies_tensor.max().item()

        # Save to CSV
        try:
            pd.DataFrame(
                {
                    "worst_performance": [worst_performance],
                    "best_performance": [best_performance],
                }
            ).to_csv(metafeatures_path, index=False)
        except:
            pass

        return worst_performance, best_performance

    def get_num_classes(self) -> int:
        # This operation is lazy and only computes the unique values when necessary
        unique_classes = self._labels["label"].drop_duplicates().compute()
        return len(unique_classes)

    def get_num_samples(self) -> int:
        # Assuming 'datapoint_id' is a column and not an index level in Dask
        unique_datapoints = self._labels["datapoint_id"].drop_duplicates().compute()
        return len(unique_datapoints)

    def get_features(self, ensembles: list[list[int]]) -> torch.Tensor:
        # Flatten the list of lists to get all model IDs
        all_model_ids = set(id for sublist in ensembles for id in sublist)
        
        # Query all needed model_ids at once
        all_needed_features = self._features[self._features['model_id'].isin(all_model_ids)].compute()
    
        features = []
        for model_ids in ensembles:
            ensemble_features = []
            for model_id in model_ids:
                matched_rows = all_needed_features[all_needed_features['model_id'] == model_id]
                ensemble_features.append(matched_rows)
            ensemble_features_df = pd.concat(ensemble_features, ignore_index=True)
            temp_features_tensor = torch.tensor(ensemble_features_df.drop(columns='model_id').values, dtype=torch.float32)
            features.append(temp_features_tensor.unsqueeze(0))

        features_tensor = torch.cat(features, axis=0)
        return features_tensor


    def get_targets(self) -> torch.Tensor:
        # Efficiently get labels and compute them to a tensor
        labels_np = self._labels["label"].compute().values  # Compute only when needed
        targets_tensor = torch.tensor(labels_np).long()

        return targets_tensor

    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:
        # Fetch probabilities using Dask
        y_proba_da = self._get_probabilities(ensembles=ensembles)
        # Convert the Dask array to a NumPy array and then to a Torch tensor
        y_proba_np = y_proba_da.compute()  # This should have shape (B, M, N, C)
        y_proba_tensor = torch.from_numpy(y_proba_np).float()  # Convert to float tensor

        # Ensure the tensor is in the correct shape, although it should already be correct
        return y_proba_tensor

    def _get_probabilities(self, ensembles: list[list[int]]) -> da.Array:
        num_datapoints = self.get_num_samples()
        num_classes = self.get_num_classes()

        ensemble_arrays = []
        for ensemble in ensembles:
            model_predictions = []
            for model_id in ensemble:
                model_preds = self._predictions[
                    self._predictions["model_id"] == model_id
                ].compute()
                model_preds = model_preds.sort_values("datapoint_id")
                model_preds_array = model_preds.drop(
                    columns=["model_id", "datapoint_id"], errors="ignore"
                ).to_numpy()
                model_predictions.append(model_preds_array)

            # Stack model predictions to ensure shape is (M, N, C)
            model_predictions_stack = np.stack(model_predictions, axis=0)

            # Check and reshape if necessary to ensure dimensions are (1, M, N, C)
            if model_predictions_stack.ndim == 3:  # This happens when M is 1
                model_predictions_stack = model_predictions_stack.reshape(
                    (1,) + model_predictions_stack.shape
                )

            # Convert the numpy array directly to a Dask array with explicit chunking
            ensemble_da = da.from_array(
                model_predictions_stack,
                chunks=(1, len(ensemble), num_datapoints, num_classes),
            )
            ensemble_arrays.append(ensemble_da)

        # Concatenate all ensemble arrays along a new batch axis
        y_proba = da.concatenate(ensemble_arrays, axis=0)
        return y_proba

    def get_time(self, ensembles: list[list[int]]) -> torch.Tensor:
        return torch.zeros(len(ensembles), len(ensembles[0]))

    def get_num_pipelines(self) -> int:
        return self.data_version_map[self.data_version]


