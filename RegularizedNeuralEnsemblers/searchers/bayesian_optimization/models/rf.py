from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from ....samplers.base_sampler import BaseSampler
from ....utils.common import move_to_device
from .base_model import BaseModel
from .utils import ConfigurableMeta


class BootstrapRandomForest(BaseModel, metaclass=ConfigurableMeta):
    default_config = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    }

    def __init__(
        self,
        sampler: BaseSampler,
        checkpoint_path: str | None = None,
        device: torch.device = torch.device("cpu"),
        ########################################
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str = 1.0,
    ):
        super().__init__(sampler=sampler, checkpoint_path=checkpoint_path, device=device)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

    def fit(
        self,
        num_epochs: int = 1,
        observed_pipeline_ids: list[int] | None = None,
        max_num_pipelines: int = 10,
        batch_size: int = 16,
    ) -> float | None:
        assert (
            max_num_pipelines == 1
        ), "RandomForestModel only supports max_num_pipelines=1"
        num_epochs = 1
        return super().fit(
            num_epochs=num_epochs,
            observed_pipeline_ids=observed_pipeline_ids,
            max_num_pipelines=max_num_pipelines,
            batch_size=batch_size,
        )

    @move_to_device
    def _fit_batch(
        self,
        pipeline_hps: torch.Tensor,
        metric_per_pipeline: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        X = pipeline_hps.squeeze(dim=1).cpu().numpy()
        y = metric.cpu().numpy()
        self.model.fit(X, y)
        pred = self.model.predict(X)
        loss = mean_squared_error(y, pred)
        return torch.tensor(loss, dtype=torch.float32)

    @move_to_device
    def validate(
        self,
        pipeline_hps: torch.Tensor,
        metric_per_pipeline: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        X = pipeline_hps.squeeze(dim=1).cpu().numpy()
        y = metric.cpu().numpy()
        pred = self.model.predict(X)
        loss = mean_squared_error(y, pred)
        return torch.tensor(loss, dtype=torch.float32)

    @move_to_device
    def predict(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        X = x.squeeze(dim=1).cpu().numpy()
        pred = np.array([tree.predict(X) for tree in self.model.estimators_]).T
        pred_mean = np.mean(pred, axis=1)
        pred_var = (pred - pred_mean.reshape(-1, 1)) ** 2
        pred_std = np.sqrt(np.mean(pred_var, axis=1))

        return torch.tensor(pred_mean, dtype=torch.float32), torch.tensor(
            pred_std, dtype=torch.float32
        )
