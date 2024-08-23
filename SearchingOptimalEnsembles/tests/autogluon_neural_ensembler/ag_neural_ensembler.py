from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


class AutoGluonNeuralEnsembler(AbstractModel):
    _oof_prediction_features: list[str] | None = None
    """List of out-of-fold prediction features in the stacking data."""
    _base_models: list[str] | None = None
    """List of base model in the stacking data."""

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> list[np.ndarray]:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._oof_prediction_features = self.feature_metadata.get_features(required_special_types=["stack"])

            if self.problem_type == MULTICLASS:
                self._base_models = list({x.rsplit("_", 1)[0] for x in self._oof_prediction_features})
            elif self.problem_type in [BINARY, REGRESSION]:
                self._base_models = self._oof_prediction_features
            else:
                raise ValueError(f"Invalid problem_type: {self.problem_type}")

            # Guard against randomness in feature order
            self._base_models = sorted(self._base_models)

        # Create list of out-of-fold predictions for each base model
        bm_preds: list[np.ndarray] = []
        for bm in self._base_models:
            if self.problem_type == BINARY:
                bm_pred = X[[bm]].copy()
                bm_pred[bm + "_0"] = 1 - bm_pred[bm]
                bm_pred = bm_pred[[bm + "_0", bm]].values
            elif self.problem_type == MULTICLASS:
                all_f_for_bm = [f for f in self._oof_prediction_features if f.startswith(f"{bm}_")]
                assert (
                    len(all_f_for_bm) == self.num_classes
                ), f"Incorrect number of out-of-fold predictions found for base model: {bm}!"
                bm_pred = X[all_f_for_bm].values
            elif self.problem_type == REGRESSION:
                bm_pred = X[[bm]].values
            else:
                raise ValueError(f"Invalid problem_type: {self.problem_type}")

            bm_preds.append(bm_pred)

        return bm_preds

    def _fit(
        self,
        X: pd.DataFrame,  # training data
        y: pd.Series,  # training labels
        num_gpus: int = 0,
        **kwargs,
    ):
        import random

        import torch
        from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler

        X = self.preprocess(X, is_train=True)
        device = "cuda" if num_gpus != 0 else "cpu"
        device = torch.device(device)

        # Get config index to use for this fit.
        params = self._get_model_params()

        # Set random seed
        random_seed = params.pop("random_state", 0)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Fit Model
        self.model = NeuralEnsembler(device=device, prediction_device=device, **params)
        self.model.fit(X, y.values)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        y_pred = self.model.predict(X).cpu().detach().numpy()  # does not have a concept of predict proba

        if self.problem_type in [REGRESSION]:
            return y_pred

        return self._convert_proba_to_unified_form(y_pred)

    def _set_default_params(self):
        default_params = {"random_state": 0, "ne_net_mode": "combined"}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            problem_types=[BINARY, MULTICLASS, REGRESSION],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params