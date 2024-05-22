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
        ne_eval_context_size: int = 50,
        ne_num_layers: int = 1,
        ne_use_context: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)

        self.hidden_dim = ne_hidden_dim
        self.context_size = ne_context_size
        self.reg_term_div = ne_reg_term_div
        self.device = device
        self.add_y = ne_add_y
        self.eval_context_size = ne_eval_context_size
        self.num_layers = ne_num_layers
        self.use_context = ne_use_context

    def set_state(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.metadataset = metadataset
        self.device = device

    def batched_prediction(self, X, base_functions,
                            y=None,
                            X_context=None,
                            y_context=None,
                            mask_context=None):
        _, num_samples, num_classes, num_pipelines = X.shape
        idx = np.arange(num_samples)
        #np.random.shuffle(idx)
        outputs = []
        weights = []

        for i in range(0, num_samples, self.eval_context_size):
            range_idx = idx[range(i, min(i + self.eval_context_size, num_samples))]

            if mask_context is not None:
                temp_mask = mask_context[:(self.context_size+len(range_idx)),
                                        : (self.context_size+len(range_idx))]
            else:
                temp_mask = None
            temp_y = y[:, range_idx] if y is not None else None
            output, w = self.net(x=X[:, range_idx], 
                                 base_functions=base_functions[:, range_idx], 
                                 y=temp_y,
                                 X_context=X_context,
                                 y_context=y_context,
                                 mask_context=temp_mask)
            w = w.transpose(2, 3).transpose(
                1, 2
            )  # Expected shape [BATCH SIZE X NUM_PIPELINES X NUM_SAMPLES X NUM_CLASSES]
            outputs.append(output)
            weights.append(w)

        return torch.cat(outputs, axis=-2).to(self.device), torch.cat(
            weights, axis=-2
        ).to(self.device)

    def get_weights(self, X_obs, 
                    X_context=None, 
                    y_context=None):
        base_functions = (
            self.metadataset.get_predictions([X_obs])[0]
            .transpose(0, 1)
            .transpose(2, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        if X_context is None or y_context is None:
            mask_context = None
        else:
            mask_context = self.get_mask_context()
        _, weights = self.batched_prediction(
            X=base_functions, base_functions=base_functions, 
            X_context=X_context, y_context=y_context, mask_context=mask_context
        )
        return weights
    
    def get_mask_context(self):
        mask = None
        if self.use_context:
            actual_eval_context_size = min(self.eval_context_size, self.metadataset.get_num_samples())
            mask_upper_left_tile = torch.ones(self.context_size, self.context_size)
            mask_upper_right_tile = torch.zeros(self.context_size, actual_eval_context_size)
            mask_lower_left_tile = torch.ones(actual_eval_context_size, self.context_size)
            mask_lower_right_tile = torch.eye(actual_eval_context_size)
            #mask_lower_right_tile = torch.ones(actual_eval_context_size, actual_eval_context_size)
            mask_upper = torch.cat([mask_upper_left_tile, mask_upper_right_tile], dim=1)
            mask_lower = torch.cat([mask_lower_left_tile, mask_lower_right_tile], dim=1)
            mask = torch.cat([mask_upper, mask_lower], dim=0).to(self.device)
            mask = mask.bool()
        return mask

    def get_context(self, X_obs, metadataset):
        #X_context are the base functions for val dataset for a subset fo samples
        #y_context are the 
        X_context = None
        y_context = None
        if self.use_context:
            base_functions = (
                metadataset.get_predictions([X_obs])[0]
                .transpose(0, 1)
                .transpose(2, 1)
                .unsqueeze(0)
            )
            _, num_samples, num_classes, num_pipelines = base_functions.shape
            samples_idx_for_context = np.random.randint(0, num_samples, self.context_size)
            X_context = base_functions[:,samples_idx_for_context].to(self.device)
            y_context = (
                metadataset.targets[samples_idx_for_context]
                .unsqueeze(0)
                .to(self.device)
            )
        
        return X_context, y_context
   
        

    def sample(self, X_obs, **kwargs) -> tuple[list, float]:
        """Fit neural ensembler, output ensemble WITH weights"""
        best_ensemble = None
        weights = None
        base_functions = (
            self.metadataset.get_predictions([X_obs])[0]
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
            num_layers=self.num_layers
        )

        criterion = nn.CrossEntropyLoss()
        y_train = torch.tensor(y_train, dtype=torch.long)
        _, num_samples, num_classes, num_base_functions = X_train.shape
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
                output.reshape(-1, num_classes), y_train[:, context_idx].reshape(-1)
            )
            div = div_loss(w, base_functions_train[:, context_idx])
            loss -= self.reg_term_div * div
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
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
        num_layers=3,
        simple_coefficients=False,
        dropout_rate=0,
        num_heads=1,
        w_norm_type="softmax",
        add_y=True,
        mask_prob=0.5,
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.mask_prob = mask_prob

        self.embedding = nn.Linear(input_dim, hidden_dim)
        #encoder_layer = nn.TransformerEncoderLayer(
        #    d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim
        #
        # )
        first_encoder_modules = [
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
            ) for _ in range(num_layers)
        ]
        #self.first_encoder = nn.Sequential(*first_encoder_modules)
        self.first_encoder = nn.ModuleList(first_encoder_modules)
        #self.first_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        if add_y:
            self.second_encoder = nn.TransformerEncoderLayer(
                d_model=hidden_dim + input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            self.out_layer = nn.Linear(hidden_dim + input_dim, output_dim)
        else:
            self.second_encoder = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                batch_first=True
            )

            self.out_layer = nn.Linear(hidden_dim, output_dim)

        self.simple_coefficients = simple_coefficients
        self.w_norm_type = w_norm_type
        self.dropout_rate = dropout_rate
        custom_weights_fc1 = torch.randn(
            output_dim
        )  # Custom weights for the first fully connected layer
        self.weight = nn.Parameter(custom_weights_fc1)
        self.dropout = nn.Dropout(p=self.dropout_rate)


    def get_max_prob_per_base_function(self, x, y):
        batch_size, num_samples, num_classes, num_base_functions = x.shape

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
        return max_prob_per_base_function

    def forward(self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None):

        if self.simple_coefficients:
            if len(base_functions.shape) == 3:
                x = torch.repeat_interleave(
                    self.weight.reshape(1, 1, -1), base_functions.shape[0], dim=0
                )
                x = torch.repeat_interleave(x, base_functions.shape[1], dim=1)
            else:
                # x = torch.repeat_interleave(self.weight.reshape(-1,1), x.shape[0], dim=1).T
                x = torch.repeat_interleave(
                    self.weight.reshape(1, -1), base_functions.shape[0], dim=0
                )

        else:
            max_prob_per_base_function = self.get_max_prob_per_base_function(x,y)
            num_query_samples = x.shape[1]

            if X_context is not None:
                max_prob_per_base_function_context = self.get_max_prob_per_base_function(X_context,y_context)
                
                if max_prob_per_base_function is not None:
                    max_prob_per_base_function = torch.cat([
                        max_prob_per_base_function_context,
                        max_prob_per_base_function
                    ], dim=1)
                x = torch.cat([
                    X_context,
                    x
                ], dim=1)
            batch_size, num_samples, num_classes, num_base_functions = x.shape

            # X = [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
            #x = torch.nn.functional.softmax(x, dim=-2)
            
            x = x.transpose(1, 2).reshape(-1, num_samples, num_base_functions)
            # [(BATCH SIZE X NUMBER OF CLASSES) X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS]
            x = self.embedding(x)
            for encoder in self.first_encoder:
                x = encoder(x, src_mask=mask_context)
            x = x.reshape(batch_size, num_classes, num_samples, self.hidden_dim)
            x = x.mean(axis=1)

            if max_prob_per_base_function is not None:
                x = torch.cat([x, max_prob_per_base_function], axis=-1)
            x = self.second_encoder(x, src_mask=mask_context)
            x = self.out_layer(x)

            if len(base_functions.shape) == 3:
                x = torch.repeat_interleave(
                    x.unsqueeze(1), base_functions.shape[1], dim=1
                )
            if len(base_functions.shape) == 4:
                x = torch.repeat_interleave(
                    x.unsqueeze(2), base_functions.shape[2], dim=2
                )

        if self.w_norm_type == "linear":
            w = nn.ReLU()(x)
            w_norm = torch.divide(w + 1e-8, torch.sum(w, axis=-1).reshape(-1, 1) + 1e-8)
        elif self.w_norm_type == "softmax":
            w = x
            w_norm = torch.nn.functional.softmax(w, dim=-1)
        else:
            raise ValueError("w_norm_type must be either linear or softmax")

        w_norm = w_norm[:,-num_query_samples:]

        x = torch.multiply(base_functions, w_norm).sum(
            axis=-1
        )  # [BATCH_SIZE, NUM_SAMPLES, NUM_CLASSES]
        return x, w_norm
