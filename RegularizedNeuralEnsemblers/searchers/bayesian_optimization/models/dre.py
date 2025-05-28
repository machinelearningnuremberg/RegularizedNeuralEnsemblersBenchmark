from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau

from ....samplers.base_sampler import BaseSampler
from ..modules.rank_loss import RankLoss
from ..modules.set_transformer import SetTransformer
from .base_model import BaseModel
from .utils import ConfigurableMeta


class DRE(BaseModel, metaclass=ConfigurableMeta):
    default_config = {
        "hidden_dim": 64,
        "hidden_dim_ff": 32,
        "num_heads": 4,
        "num_seeds": 1,
        "out_dim": 32,
        "out_dim_ff": 1,
        "num_encoders": 4,
        "num_layers_ff": 1,
        "add_y": True,
        "criterion_type": "weighted_listwise",
        "lr": 1e-3,
    }

    def __init__(
        self,
        sampler: BaseSampler,
        add_y: bool = True,
        checkpoint_path: Path | None = None,
        device: torch.device = torch.device("cpu"),
        #############################################
        hidden_dim: int = 64,
        hidden_dim_ff: int = 128,
        num_heads: int = 4,
        num_seeds: int = 1,
        out_dim: int = 32,
        out_dim_ff: int = 1,
        num_encoders: int = 2,
        num_layers_ff: int = 1,
        num_context_pipelines: int = 10,
        criterion_type: str = "weighted_listwise",
        activation_output: str = "sigmoid",
        score_with_rank: bool = False,
        lr: float = 1e-3,
    ):
        super().__init__(
            sampler=sampler, add_y=add_y, checkpoint_path=checkpoint_path, device=device
        )

        assert num_encoders > 0, "num_encoders must be greater than 0"
        assert num_layers_ff > 0, "num_layers_ff must be greater than 1"

        self.activation_output = activation_output
        self.score_with_rank = score_with_rank

        self.encoder = SetTransformer(
            self.dim_in, hidden_dim, num_heads, num_seeds, out_dim
        )
        self.num_encoders = num_encoders
        self.num_context_pipelines = num_context_pipelines

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            nn.Linear(in_features=out_dim * num_encoders, out_features=hidden_dim_ff)
        )
        for _ in range(1, num_layers_ff):
            self.hidden_layers.append(
                nn.Linear(in_features=hidden_dim_ff, out_features=hidden_dim_ff)
            )

        self.out_layer = nn.ModuleList()
        for _ in range(num_encoders):
            self.out_layer.append(
                nn.Linear(in_features=hidden_dim_ff, out_features=out_dim_ff)
            )

        self.device = sampler.device  # Why is it needed?
        self.criterion = RankLoss(type=criterion_type)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def _fit_batch(
        self,
        pipeline_hps: torch.Tensor,
        metric_per_pipeline: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        # pylint: disable=unused-variable
        batch_size, num_pipelines, _ = pipeline_hps.shape
        batches = [
            self.sampler.sample(
                fixed_num_pipelines=self.num_context_pipelines,
                batch_size=batch_size,
                observed_pipeline_ids=self.observed_pipeline_ids,
            )
            for _ in range(self.num_encoders - 1)
        ]
        X = [pipeline_hps]
        y_e = [metric]
        y_p = [metric_per_pipeline]

        for batch in batches:
            X.append(batch[0])
            y_e.append(batch[1])
            y_p.append(batch[2])

        self.optimizer.zero_grad()
        y_pred = self.forward(X, y_p)
        loss = torch.Tensor([0]).cuda()

        k = 0
        for y_pred_, y_e_ in zip(y_pred, y_e):
            loss += (
                self.criterion(y_pred_.reshape(y_e_.shape), y_e_)
            ) / self.num_encoders

            k += kendalltau(y_pred_.detach().cpu().numpy(), y_e_.detach().cpu().numpy())[
                0
            ]
            if np.isnan(k):
                self.logger.debug("Kendall tau is nan!!!")

        self.logger.debug(f"Avg. Kendall tau: {k / len(y_pred)}")
        loss.backward()
        self.optimizer.step()

        return loss

    def forward(self, x, y):
        assert len(x) == self.num_encoders
        assert len(y) == self.num_encoders

        y = y.copy()
        x_agg = []

        for i in range(self.num_encoders):
            if self.add_y:
                with torch.no_grad():
                    if self.training:
                        y[i], mask = self._mask_y(y[i], x[i].shape)
                    else:
                        # mask = torch.ones(y[i].shape).bool().to(x[i].device)
                        mask = ~torch.isnan(y[i]).unsqueeze(-1)
                        y[i] = y[i].to(x[i].device).unsqueeze(-1)
                        y[i][~mask] = 0
                    mean = y[i][mask].mean()
                    std = y[i][mask].std()
                    if torch.isnan(std) or std == 0:
                        std = 1
                    y[i] = (y[i] - mean) / std
                    temp_x = torch.cat([x[i], y[i], mask.int()], axis=-1)
            else:
                temp_x = x[i]
            temp_x = self.encoder(temp_x)
            x_agg.append(temp_x)

        x = torch.cat(x_agg, axis=-1)
        x = self.hidden_layers[0](x)

        for layer in self.hidden_layers[1:]:
            x = nn.ReLU()(x)
            x = layer(x)
        x = nn.ReLU()(x)
        # x = nn.Sigmoid()(x)

        if self.activation_output == "sigmoid":
            out = [nn.Sigmoid()(f(x)) for f in self.out_layer]
        else:
            out = [f(x) for f in self.out_layer]
        return out

    def predict(
        self,
        x,
        metric_per_pipeline: torch.Tensor = None,
        score_with_rank: bool = False,  # pylint: disable=unused-argument
        max_num_pipelines: int | None = None,  # pylint: disable=unused-argument
        **kwargs,
    ):
        self.eval()

        with torch.no_grad():
            batch_size, num_pipelines, _ = x.shape  # pylint: disable=unused-variable
            batches = [
                self.sampler.sample(
                    fixed_num_pipelines=self.num_context_pipelines,
                    batch_size=batch_size,
                    observed_pipeline_ids=self.observed_pipeline_ids,
                )
                for _ in range(self.num_encoders - 1)
            ]
            X = [x]
            y_p = [metric_per_pipeline]

            for batch in batches:
                X.append(batch[0])
                y_p.append(batch[2])

            out_list = []
            for i in range(self.num_encoders):
                x_temp, y_temp = X.copy(), y_p.copy()
                x_temp[i] = X[0]
                y_temp[i] = y_p[0]
                x_temp[0] = X[i]
                y_temp[0] = y_p[i]

                pred = self.forward(x_temp, y_temp)[i].squeeze(-1).squeeze(-1)

                if self.score_with_rank:
                    scores = self.get_rank(pred)
                else:
                    scores = pred
                out_list.append(scores)

            out_mean = torch.mean(torch.stack(out_list), axis=0)
            out_std = torch.std(torch.stack(out_list), axis=0)

        return out_mean, out_std

    def get_rank(self, x, descending=False):
        # x += torch.rand(x.shape).to(x.device) * 1e-5

        # x += torch.rand(x.shape).to(x.device) * 1e-5
        sorted_indices = torch.argsort(x, descending=descending)  #
        ranks = torch.zeros_like(x).to(x.device)
        # Assign ranks to each element based on their sorted indices
        ranks[sorted_indices] = torch.arange(len(x)).to(x.device).float()

        # TODO: use rankdata from scipy
        # ranks = scipy.stats.rankdata(x, axis=-1)

        return ranks

    def validate(self, pipeline_hps, metric_per_pipeline, metric):
        batch_size, num_pipelines, _ = pipeline_hps.shape
        batches = [
            self.sampler.sample(fixed_num_pipelines=num_pipelines, batch_size=batch_size)
            for _ in range(self.num_encoders - 1)
        ]
        X = [pipeline_hps]
        y_e = [metric]
        y_p = [metric_per_pipeline]

        for batch in batches:
            X.append(batch[0])
            y_e.append(batch[1])
            y_p.append(batch[2])

        self.eval()
        y_pred = self.forward(X, y_p)
        loss = self.criterion(y_pred[0].reshape(metric.shape), metric)
        self.train()

        return loss

    def save_checkpoint(self, checkpoint: Path = Path("surrogate.pth")):
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint,
        )

    def load_checkpoint(self, checkpoint: Path = Path("surrogate.pth")):
        ckpt = torch.load(checkpoint, map_location=torch.device(self.device))

        self.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
