import torch
import torch.nn as nn

from ..modules.set_transformer import SetTransformer
from .base_model import BaseModel


class DRE(BaseModel):
    def __init__(
        self,
        sampler,
        checkpoint_path,
        device,
        dim_in,
        hidden_dim=64,
        hidden_dim_ff=32,
        num_heads=4,
        num_seeds=1,
        out_dim=32,
        out_dim_ff=1,
        num_encoders=2,
        num_layers_ff=1,
        add_y=False,
    ):
        super().__init__(sampler, checkpoint_path, device)

        assert num_encoders > 0, "num_encoders must be greater than 0"
        assert num_layers_ff > 0, "num_layers_ff must be greater than 1"

        self.add_y = add_y
        if add_y:
            dim_in += 1

        self.encoder = SetTransformer(dim_in, hidden_dim, num_heads, num_seeds, out_dim)
        self.num_encoders = num_encoders

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            nn.Linear(in_features=out_dim * num_encoders, out_features=hidden_dim_ff)
        )
        for i in range(1, num_layers_ff):
            self.hidden_layers.append(
                nn.Linear(in_features=hidden_dim_ff, out_features=hidden_dim_ff)
            )

        self.out_layer = nn.ModuleList()
        for i in range(num_encoders):
            self.out_layer.append(
                nn.Linear(in_features=hidden_dim_ff, out_features=out_dim_ff)
            )

    def mask_y(self, y, shape, device):
        if y is None:
            y = torch.zeros(shape[0], shape[1], 1).to(device)
            mask = y
        else:
            ones_pct = 1 - 1 / shape[1]
            y_temp = y.unsqueeze(-1)
            mask = torch.bernoulli(torch.full(y_temp.shape, ones_pct)).to(device)

            if torch.sum(mask) == 0:
                mask = torch.ones(mask.shape).to(device)

            y_temp = y_temp.to(device)
            y = y_temp * mask
        return y, mask.bool()

    def forward(self, x, y):
        assert len(x) == self.num_encoders
        assert len(y) == self.num_encoders

        y = y.copy()
        x_agg = []

        for i in range(self.num_encoders):
            if self.add_y:
                with torch.no_grad():
                    if self.training:
                        y[i], mask = self.mask_y(y[i], x[i].shape, x[i].device)
                    else:
                        mask = torch.ones(y[i].shape).bool().to(x[i].device)
                        y[i] = y[i].to(x[i].device).unsqueeze(-1)
                    mean = y[i][mask].mean()
                    std = y[i][mask].std()
                    if torch.isnan(std) or std == 0:
                        std = 1
                    y[i] = (y[i] - mean) / std
                    temp_x = torch.cat([x[i], y[i]], axis=-1)
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
        out = [f(x) for f in self.out_layer]

        return out

    def predict(self, x, y):
        with torch.no_grad():
            x_query = x[0]
            y_query = y[0]
            out_list = []
            for i in range(self.num_encoders):
                x_temp, y_temp = x, y
                x_temp[i] = x_query
                y_temp[i] = y_query
                x_temp[0] = x[i]
                y_temp[0] = y[i]
                ranks = self.get_rank(self.forward(x, y)[i].squeeze(-1).squeeze(-1))
                out_list.append(ranks)
            out_mean = torch.mean(torch.stack(out_list), axis=0)
            out_std = torch.std(torch.stack(out_list), axis=0)
        return out_mean, out_std

    def get_rank(self, x):
        # x += torch.rand(x.shape).to(x.device) * 1e-5
        sorted_indices = torch.argsort(x)

        # Create a tensor to store the ranks
        ranks = torch.zeros_like(x).to(x.device)

        # Assign ranks to each element based on their sorted indices
        ranks[sorted_indices] = torch.arange(len(x)).to(x.device).float()
        return ranks

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
