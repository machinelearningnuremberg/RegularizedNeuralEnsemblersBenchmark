from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..samplers.random_sampler import RandomSampler
from .base_ensembler import BaseEnsembler


def div_loss(w, base_observations):
    criterion = nn.KLDivLoss(reduction="none")
    batch_size, num_samples, num_classes, num_pipelines = base_observations.shape
    pred = torch.multiply(w, base_observations).transpose(2, 3).reshape(-1, num_classes)
    loss = criterion(
        pred, base_observations.transpose(2, 3).reshape(-1, num_classes)
    ).mean(-1)
    loss = loss.reshape(batch_size, num_samples, num_pipelines)
    div = torch.multiply(loss, w[:, :, 0, :].squeeze(-1))
    return div.mean()


class NeuralEnsembler(BaseEnsembler):
    """Neural (End-to-End) Ensembler."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cuda"),
        ne_hidden_dim: int = 512,
        ne_context_size: int = 32,
        ne_reg_term_div: float = 0.1,
        ne_add_y: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)

        self.hidden_dim = ne_hidden_dim
        self.context_size = ne_context_size
        self.reg_term_div = ne_reg_term_div
        self.device = device
        self.add_y = ne_add_y

    def set_state(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.metadataset = metadataset
        self.device = device

    def batched_prediction(self, X, base_functions, y=None):
        _, num_samples, num_classes, num_pipelines = X.shape
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        outputs = []
        weights = []

        for i in range(0, num_samples, self.context_size):
            range_idx = idx[range(i, min(i + self.context_size, num_samples))]
            temp_y = y[:, range_idx] if y is not None else None
            output, w = self.net(X[:, range_idx], base_functions[:, range_idx], y=temp_y)
            w = w.transpose(2, 3).transpose(
                1, 2
            )  # Expected shape [BATCH SIZE X NUM_PIPELINES X NUM_SAMPLES X NUM_CLASSES]
            outputs.append(output)
            weights.append(w)

        return torch.cat(outputs, axis=-2).to(self.device), torch.cat(
            weights, axis=-2
        ).to(self.device)

    def get_weights(self, X_obs):
        base_functions = (
            self.metadataset.predictions[X_obs]
            .transpose(0, 1)
            .transpose(2, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        _, weights = self.batched_prediction(
            X=base_functions, base_functions=base_functions
        )
        return weights

    def sample(self, X_obs, **kwargs) -> tuple[list, float]:
        """Fit neural ensembler, output ensemble WITH weights"""
        best_ensemble = None
        weights = None
        base_functions = (
            self.metadataset.predictions[X_obs]
            .transpose(0, 1)
            .transpose(2, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        y = self.metadataset.targets.unsqueeze(0).to(self.device)
        self.net = self.fit_net(
            X_train=base_functions, y_train=y, base_functions_train=base_functions
        )
        _, weights = self.batched_prediction(
            X=base_functions, base_functions=base_functions, y=y
        )
        best_ensemble = X_obs.tolist()
        _, best_metric, _, _ = self.metadataset.evaluate_ensembles_with_weights(
            [best_ensemble], weights
        )

        return best_ensemble, best_metric

    def send_to_device(self, *args):
        output = []
        for arg in args:
            output.append(arg.to(self.device))
        return output

    def fit_net(
        self,
        X_train,
        y_train,
        base_functions_train,
        simple_coefficients=False,
        learning_rate=0.0001,
        epochs=1000,
        w_norm_type="softmax",
        dropout_rate=0,
    ):
        output_dim = base_functions_train.shape[
            -1
        ]  # [NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS]
        input_dim = X_train.shape[-1]

        net = ENet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            simple_coefficients=simple_coefficients,
            w_norm_type=w_norm_type,
            dropout_rate=dropout_rate,
            add_y=self.add_y,
        )

        criterion = nn.CrossEntropyLoss()
        y_train = torch.tensor(y_train, dtype=torch.long)
        _, num_samples, num_classes, num_pipelines = X_train.shape
        idx = np.arange(num_samples)
        optimizer = Adam(net.parameters(), lr=learning_rate)
        net.train()

        X_train, y_train, base_functions_train, net = self.send_to_device(
            X_train, y_train, base_functions_train, net
        )

        for epoch in range(epochs):
            optimizer.zero_grad()
            np.random.shuffle(idx)
            context_idx = idx[: self.context_size]
            output, w = net(
                X_train[:, context_idx],
                base_functions_train[:, context_idx],
                y_train[:, context_idx],
            )
            loss = criterion(
                output.reshape(-1, num_classes), y_train.reshape(-1)[context_idx]
            )
            div = div_loss(w, base_functions_train[:, context_idx])
            loss -= self.reg_term_div * div
            loss.backward()
            optimizer.step()
            print("Epoch", epoch, "Loss", loss.item(), "Div Loss", div.item())

        net.eval()
        return net


class ENet(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        output_dim=1,
        num_layers=2,
        simple_coefficients=False,
        dropout_rate=0,
        num_heads=1,
        num_encoders=1,
        w_norm_type="softmax",
        add_y=True,
        mask_prob=1.0,
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoders = num_encoders
        self.add_y = add_y
        self.mask_prob = mask_prob

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )

        self.first_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        if add_y:
            self.second_encoder = nn.TransformerEncoderLayer(
                d_model=hidden_dim + input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
            )
            self.out_layer = nn.Linear(hidden_dim + input_dim, output_dim)
        else:
            self.second_encoder = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim
            )

            self.out_layer = nn.Linear(hidden_dim, output_dim)

        if self.num_layers == 1:
            self.layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)]
            )

        else:
            self.layers = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_features=1))

        self.simple_coefficients = simple_coefficients
        self.w_norm_type = w_norm_type
        self.dropout_rate = dropout_rate
        custom_weights_fc1 = torch.randn(
            output_dim
        )  # Custom weights for the first fully connected layer
        self.weight = nn.Parameter(custom_weights_fc1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, base_functions_, y=None):
        if self.simple_coefficients:
            if len(base_functions_.shape) == 3:
                x = torch.repeat_interleave(
                    self.weight.reshape(1, 1, -1), base_functions_.shape[0], dim=0
                )
                x = torch.repeat_interleave(x, base_functions_.shape[1], dim=1)
            else:
                # x = torch.repeat_interleave(self.weight.reshape(-1,1), x.shape[0], dim=1).T
                x = torch.repeat_interleave(
                    self.weight.reshape(1, -1), base_functions_.shape[0], dim=0
                )

        else:
            # X = [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
            batch_size, num_samples, num_classes, num_base_functions = x.shape
            x = torch.nn.functional.softmax(x, dim=-2)

            if y is not None and self.add_y:
                with torch.no_grad():
                    y_rep = torch.repeat_interleave(y.unsqueeze(-1), x.shape[2], dim=2)
                    y_rep = torch.repeat_interleave(
                        y_rep.unsqueeze(-1), x.shape[3], dim=3
                    )
                    max_prob_per_base_function = torch.gather(
                        x.clone().detach(), 2, y_rep
                    )[:, :, 0, :].to(x.device)

                    if self.training:
                        mask = torch.randn(
                            batch_size, num_samples, num_base_functions
                        ).to(x.device)
                        mask = (torch.rand(mask.shape) > self.mask_prob).to(x.device)
                        max_prob_per_base_function = torch.where(
                            mask,
                            max_prob_per_base_function,
                            torch.tensor(-1.0).to(x.device),
                        ).to(x.device)
            else:
                if self.add_y:
                    max_prob_per_base_function = -1 * torch.ones(
                        batch_size, num_samples, num_base_functions
                    ).to(x.device)
                else:
                    max_prob_per_base_function = None

            x = x.transpose(1, 2).reshape(-1, num_samples, num_base_functions)
            # [(BATCH SIZE X NUMBER OF CLASSES) X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS]
            x = self.embedding(x)
            x = self.first_encoder(x)
            x = x.reshape(batch_size, num_classes, num_samples, self.hidden_dim)
            x = x.mean(axis=1)

            if max_prob_per_base_function is not None:
                x = torch.cat([x, max_prob_per_base_function], axis=-1)
            x = self.second_encoder(x)
            x = self.out_layer(x)

            if len(base_functions_.shape) == 3:
                x = torch.repeat_interleave(
                    x.unsqueeze(1), base_functions_.shape[1], dim=1
                )
            if len(base_functions_.shape) == 4:
                x = torch.repeat_interleave(
                    x.unsqueeze(2), base_functions_.shape[2], dim=2
                )

        if self.w_norm_type == "linear":
            w = nn.ReLU()(x)
            w_norm = torch.divide(w + 1e-8, torch.sum(w, axis=-1).reshape(-1, 1) + 1e-8)
        elif self.w_norm_type == "softmax":
            w = x
            w_norm = torch.nn.functional.softmax(w, dim=-1)
        else:
            raise ValueError("w_norm_type must be either linear or softmax")

        x = torch.multiply(base_functions_, w_norm).sum(
            axis=-1
        )  # [BATCH_SIZE, NUM_SAMPLES, NUM_CLASSES]
        return x, w_norm
