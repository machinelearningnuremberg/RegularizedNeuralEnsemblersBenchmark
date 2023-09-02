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
        "ard": False,
        "nu": 2.5,
        "lr": 1e-3,
    }

    def __init__(
        self,
        sampler: BaseSampler,
        checkpoint_path: Path | None = None,
        device: torch.device = torch.device("cpu"),
        #############################################
        kernel_name: str = "matern",
        ard: bool = False,
        nu: float = 2.5,
        lr: float = 1e-3,
    ):
        super().__init__(sampler=sampler, checkpoint_path=checkpoint_path, device=device)

        dim_in = self.sampler.metadataset.feature_dim
        if dim_in is None:
            raise ValueError("Feature dimension is None")
        self.encoder = SetTransformer(dim_in=dim_in).to(self.device)

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
        train_x = torch.ones(train_size, self.encoder.dim_out).to(self.device)
        train_y = torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            dims=self.encoder.dim_out,
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

    def _fit_batch(
        self,
        pipeline_hps: torch.Tensor,
        metric_per_pipeline: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        self.model.train()
        self.encoder.train()
        self.likelihood.train()

        self.optimizer.zero_grad()
        z = self.encoder(pipeline_hps)
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
    ) -> torch.Tensor:
        self.model.eval()
        self.encoder.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z = self.encoder(pipeline_hps)
            predictions = self.model(z)

            mse = torch.nn.MSELoss()
            loss = mse(predictions.mean, metric)

        return loss

    def predict(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.encoder.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.encoder(x).detach()
            pred = self.likelihood(self.model(z_query))

        return pred.mean, pred.stddev
