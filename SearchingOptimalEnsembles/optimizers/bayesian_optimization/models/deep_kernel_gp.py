from __future__ import annotations

import logging
from pathlib import Path

import gpytorch
import torch

from ....samplers.base_sampler import BaseSampler
from ....utils.common import move_to_device
from ..modules.gp import ExactGPLayer
from ..modules.set_transformer import SetTransformer
from .base_model import BaseModel

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("fsspec").setLevel(logging.WARNING)


class DeepKernelGP(BaseModel):
    def __init__(
        self,
        sampler: BaseSampler,
        config={"kernel": "matern", "ard": False, "nu": 2.5},
        checkpoint_path: Path | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(sampler=sampler, checkpoint_path=checkpoint_path, device=device)

        dim_in = self.sampler.metadataset.feature_dim
        if dim_in is None:
            raise ValueError("Feature dimension is None")
        self.feature_extractor = SetTransformer(dim_in=dim_in).to(self.device)

        self.model, self.likelihood, self.mll = self._get_model_likelihood_mll(
            config=config
        )
        lr = 0.001
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters(), "lr": lr},
                {"params": self.feature_extractor.parameters(), "lr": lr},
            ]
        )

    @move_to_device
    def _get_model_likelihood_mll(
        self, config: dict, train_size: int = 1
    ) -> tuple[
        ExactGPLayer,
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.mlls.ExactMarginalLogLikelihood,
    ]:
        train_x = torch.ones(train_size, self.feature_extractor.dim_out).to(self.device)
        train_y = torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            config=config,
            dims=self.feature_extractor.dim_out,
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        return model, likelihood, mll

    def load_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        if self.checkpoint_path is None:
            logging.info("No checkpoint specified. Skipping...")
            return

        checkpoint = self.checkpoint_path / checkpoint_name
        logging.info(f"Loading checkpoint from {checkpoint}...")

        try:
            if not Path(checkpoint).exists():
                logging.info(f"Checkpoint {checkpoint} does not exist. Skipping...")
                return
            ckpt = torch.load(checkpoint, map_location=torch.device(self.device))
            self.model.load_state_dict(ckpt["model"])
            self.likelihood.load_state_dict(ckpt["likelihood"])
            self.feature_extractor.load_state_dict(ckpt["feature_extractor"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            logging.info(f"Exception {e}")

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        if self.checkpoint_path is None:
            logging.info("No checkpoint specified. Skipping...")
            return

        checkpoint = self.checkpoint_path / checkpoint_name
        logging.info(f"Saving checkpoint to {checkpoint}...")

        torch.save(
            {
                "model": self.model.state_dict(),
                "likelihood": self.likelihood.state_dict(),
                "feature_extractor": self.feature_extractor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint,
        )

    def _fit_batch(
        self,
        pipeline_hps: torch.Tensor,
        metric: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad()
        z = self.feature_extractor(pipeline_hps)
        self.model.set_train_data(inputs=z, targets=metric, strict=False)
        predictions = self.model(z)

        loss = -self.mll(predictions, self.model.train_targets)
        loss.backward()
        optimizer.step()

        return loss, self.likelihood.noise

    def predict(
        self, x: torch.Tensor, sampler: BaseSampler
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        # TODO: fix the concatenation
        X_obs = torch.ones(1, self.feature_extractor.dim_in).to(self.device)
        y_obs = torch.ones(1).to(self.device)
        # TODO: do from 1 to max_num_pipelines
        for num_pipelines in range(10, 12):
            # pylint: disable=unused-variable
            pipeline_hps, metric, metric_per_pipeline, time_per_pipeline = sampler.sample(
                observed_pipeline_ids=self.observed_ids,
                max_num_pipelines=num_pipelines,
            )
            y_obs = torch.cat([y_obs, metric], dim=0)
            z_support = self.feature_extractor(pipeline_hps).detach()
            X_obs = torch.cat([X_obs, z_support], dim=0)

        self.model.set_train_data(inputs=X_obs, targets=y_obs, strict=False)

        with torch.no_grad():
            z_query = self.feature_extractor(x).detach()
            pred = self.likelihood(self.model(z_query))

        return pred.mean, pred.stddev
