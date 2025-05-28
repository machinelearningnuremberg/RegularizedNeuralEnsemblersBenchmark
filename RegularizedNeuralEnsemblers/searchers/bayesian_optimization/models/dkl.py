from __future__ import annotations

from pathlib import Path

import gpytorch
import torch

from ....samplers.base_sampler import BaseSampler
from ....utils.common import move_to_device
from ..modules.gp import ExactGPLayer
from ..modules.set_transformer import SetTransformer
from .base_model import BaseModel
from .utils import ConfigurableMeta


class DeepKernelGP(BaseModel, metaclass=ConfigurableMeta):
    default_config = {
        "kernel_name": "matern",
        "ard": True,
        "nu": 1.5,
        "hidden_dim": 128,
        "out_dim": 20,
        "num_heads": 8,
        "num_seeds": 1,
        "lr": 1e-4,
        "add_y": True,
        # "optional_dim": None,
    }

    def __init__(
        self,
        sampler: BaseSampler,
        add_y: bool = True,
        checkpoint_path: Path | None = None,
        device: torch.device = torch.device("cpu"),
        #############################################
        kernel_name: str = "matern",
        ard: bool = False,
        nu: float = 2.5,
        hidden_dim: int = 64,
        out_dim: int = 20,
        num_heads: int = 4,
        num_seeds: int = 1,
        lr: float = 1e-3,
        # optional_dim: int | None = None,
    ):
        super().__init__(
            sampler=sampler, add_y=add_y, checkpoint_path=checkpoint_path, device=device
        )

        self.encoder = SetTransformer(
            dim_in=self.dim_in,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_seeds=num_seeds,
            out_dim=out_dim,
            # optional_dim=optional_dim,
        ).to(self.device)

        self.model, self.likelihood, self.mll = self._get_model_likelihood_mll(
            kernel_name=kernel_name, ard=ard, nu=nu
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters(), "lr": lr},
                {"params": self.encoder.parameters(), "lr": lr},
            ]
        )

    @move_to_device
    def _get_model_likelihood_mll(
        self, kernel_name, ard: bool, nu: float, train_size: int = 1
    ) -> tuple[
        ExactGPLayer,
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.mlls.ExactMarginalLogLikelihood,
    ]:
        train_x = torch.ones(train_size, self.encoder.out_dim).to(self.device)
        train_y = torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            dims=self.encoder.out_dim,
            kernel_name=kernel_name,
            ard=ard,
            nu=nu,
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        return model, likelihood, mll

    def load_checkpoint(self, checkpoint: Path = Path("surrogate.pth")):
        ckpt = torch.load(checkpoint, map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt["model"])
        self.likelihood.load_state_dict(ckpt["likelihood"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def save_checkpoint(self, checkpoint: Path = Path("surrogate.pth")):
        torch.save(
            {
                "model": self.model.state_dict(),
                "likelihood": self.likelihood.state_dict(),
                "encoder": self.encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint,
        )

    def _transform_input(
        self, pipeline_hps: torch.Tensor, metric_per_pipeline: torch.Tensor
    ) -> torch.Tensor:
        x = pipeline_hps
        if self.add_y:
            if self.model.training:
                metric_per_pipeline, mask = self._mask_y(
                    metric_per_pipeline, pipeline_hps.shape[:-1]
                )
            else:
                mask = ~torch.isnan(metric_per_pipeline).unsqueeze(-1)
                metric_per_pipeline = metric_per_pipeline.to(self.device).unsqueeze(-1)
                metric_per_pipeline[~mask] = 0
            x = torch.cat([pipeline_hps, metric_per_pipeline, mask.float()], dim=-1)
        return x

    def _fit_batch(
        self,
        pipeline_hps: torch.Tensor,
        metric_per_pipeline: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        self.model.train()
        self.encoder.train()
        self.likelihood.train()

        x = self._transform_input(pipeline_hps, metric_per_pipeline)

        self.optimizer.zero_grad()
        z = self.encoder(x).squeeze()
        self.model.set_train_data(inputs=z, targets=metric, strict=False)
        predictions = self.model(z)
        loss = -self.mll(predictions, self.model.train_targets)
        loss.backward()
        self.optimizer.step()

        return loss

    def validate(
        self,
        pipeline_hps: torch.Tensor,
        metric_per_pipeline: torch.Tensor,
        metric: torch.Tensor,
        max_num_pipelines: int = 10,
    ) -> torch.Tensor:
        self.observed_pipeline_ids = None
        X_obs, y_obs = self._create_support(max_num_pipelines=max_num_pipelines)

        self.model.set_train_data(inputs=X_obs, targets=y_obs, strict=False)

        self.model.eval()
        self.encoder.eval()
        self.likelihood.eval()

        with torch.no_grad():
            x = self._transform_input(pipeline_hps, metric_per_pipeline)
            z = self.encoder(x).squeeze()
            predictions = self.model(z)

            mse = torch.nn.MSELoss()
            loss = mse(predictions.mean, metric)

        return loss

    def predict(
        self,
        x: torch.Tensor,
        metric_per_pipeline: torch.Tensor = None,
        max_num_pipelines: int = 10,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.encoder.eval()
        self.likelihood.eval()

        X_obs, y_obs = self._create_support(max_num_pipelines=max_num_pipelines)
        self.model.set_train_data(inputs=X_obs, targets=y_obs, strict=False)

        with torch.no_grad():
            x = self._transform_input(x, metric_per_pipeline)
            z_query = self.encoder(x).squeeze().detach()
            pred = self.likelihood(self.model(z_query))

        return pred.mean, pred.stddev

    def _create_support(
        self, max_num_pipelines: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Initialize lists to collect the tensors
        X_obs_list = []
        y_obs_list = []
        for num_pipelines in range(1, max_num_pipelines + 1):
            (
                pipeline_hps,
                metric,
                metric_per_pipeline,
                _,
                _,
            ) = self.sampler.sample(
                observed_pipeline_ids=self.observed_pipeline_ids,
                max_num_pipelines=num_pipelines,
            )
            y_obs_list.append(metric)

            x = self._transform_input(pipeline_hps, metric_per_pipeline)
            z_support = self.encoder(x).squeeze().detach()
            X_obs_list.append(z_support)

        # Concatenate the lists to form tensors
        y_obs = torch.cat(y_obs_list, dim=0)
        X_obs = torch.cat(X_obs_list, dim=0)

        return X_obs, y_obs
