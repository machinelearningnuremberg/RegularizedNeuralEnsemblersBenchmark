# type: ignore
# pylint: skip-file

import json
import os

import numpy as np
import pandas as pd
import torch
from rank_loss import RankLoss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR

PATH = "../AutoFinetune/aft_data/predictions/"

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class EnsembleLoader:
    def __init__(
        self,
        dataset=None,
        batch_size=16,
        max_num_pipelines=10,
        split="val",
        ensemble_type="soft",
        data_pct=(0.6, 0.2, 0.2),
        return_context=False,
        return_acc=True,
        max_context_size=50,
        val_freq=50,
        device="cuda",
    ):
        self.hps = pd.read_csv(os.path.join(PATH, "preprocessed_args.csv"))

        # read json file
        # self.preprocessed_args = json.load(open(os.path.join(PATH, "preprocessed_args.json"), "r"))
        # self.aggregated_info = json.load(open(os.path.join(PATH, "aggregated_info.json"), "r"))
        self.batch_size = batch_size
        self.split = split
        self.max_num_pipelines = max_num_pipelines
        self.ensembling_type = ensemble_type
        self.hps_names = self.hps.columns
        self.device = device
        self.data_pct = data_pct
        self.return_context = return_context
        self.return_acc = return_acc
        self.max_context_size = max_context_size
        self.val_freq = val_freq

        assert self.max_num_pipelines > 0, "max_num_models must be greater than 0"

        self.aggregate_info()

        if dataset is not None:
            self.datasets = [dataset]
        else:
            self.datasets = list(self.aggregated_info.keys())
            self.datasets = self.filter_datasets(self.datasets)

            self.N = len(self.datasets)
            self.split_datasets = {
                "train": self.datasets[: int(self.N * self.data_pct[0])],
                "val": self.datasets[
                    int(self.N * self.data_pct[0]) : int(
                        self.N * (self.data_pct[0] + self.data_pct[1])
                    )
                ],
                "test": self.datasets[
                    int(self.N * (self.data_pct[0] + self.data_pct[1])) :
                ],
            }

    def filter_datasets(self, datasets):
        return [x for x in datasets if "micro" in x]

    def aggregate_info(self):
        self.aggregated_info = {}
        datasets = os.listdir(os.path.join(PATH, "per_dataset"))
        for dataset in datasets:
            # read json
            # with open(os.path.join(PATH, "per_dataset", dataset, "time_info.json"), "r") as f:
            #    time_info = json.load(f)
            time_info = None

            self.aggregated_info[dataset] = {
                "predictions": np.load(
                    os.path.join(PATH, "per_dataset", dataset, "predictions.npy")
                ),
                "targets": np.load(
                    os.path.join(PATH, "per_dataset", dataset, "targets.npy")
                ),
                "split_indicator": np.load(
                    os.path.join(PATH, "per_dataset", dataset, "split_indicator.npy")
                ),
                "time_info": time_info,
            }

    def get_context(self, dataset, mode="train", max_context_size=100):
        context_size = np.random.randint(1, max_context_size + 1)
        return self.get_batch(
            mode,
            dataset=dataset,
            max_num_pipelines=1,
            batch_size=context_size,
            return_context=False,
        )

    def get_batch(
        self,
        mode="train",
        max_num_pipelines=None,
        dataset=None,
        batch_size=None,
        return_context=True,
    ):
        if max_num_pipelines is None:
            max_num_pipelines = self.max_num_pipelines

        if batch_size is None:
            batch_size = self.batch_size

        if dataset is None:
            dataset = np.random.choice(self.split_datasets[mode])

        predictions = np.array(self.aggregated_info[dataset]["predictions"])
        max_num_pipelines = min(max_num_pipelines, predictions.shape[0])
        num_pipelines = np.random.randint(1, max_num_pipelines + 1)
        pipelines_ids = np.random.randint(
            0,
            len(self.aggregated_info[dataset]["predictions"]),
            (batch_size, num_pipelines),
        )
        is_test_id = np.array(self.aggregated_info[dataset]["split_indicator"])

        new_dataset_name = dataset.replace("_v1", "")
        pipeline_hps = self.hps[
            self.hps[f"cat__dataset_mtlbm_{new_dataset_name}"] == 1
        ].values[pipelines_ids]
        targets = np.array(self.aggregated_info[dataset]["targets"])
        if self.split == "val":
            predictions = predictions[:, is_test_id == 0, :]
            targets = targets[is_test_id == 0]
        elif self.split == "test":
            predictions = predictions[:, is_test_id == 1, :]
            targets = targets[is_test_id == 1]
        else:
            raise ValueError("split must be either val or test")

        predictions[np.isnan(predictions)] = 0
        predictions = torch.FloatTensor(predictions[pipelines_ids])
        targets = torch.unsqueeze(torch.LongTensor(targets), 0)
        targets = torch.tile(targets, (batch_size, 1))
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        acc_per_pipeline = (
            (predictions.argmax(axis=-1) == targets.unsqueeze(1)).float().mean(axis=-1)
        )
        predictions = torch.mean(predictions, axis=1)
        n_classes = predictions.shape[-1]
        ce_per_sample = cross_entropy(
            predictions.reshape(-1, n_classes), targets.reshape(-1)
        )
        ce = ce_per_sample.reshape(batch_size, -1).mean(axis=1)
        acc_per_sample = (
            predictions.reshape(-1, n_classes).argmax(axis=1) == targets.reshape(-1)
        ).float()
        acc = acc_per_sample.reshape(batch_size, -1).mean(axis=1)

        pipeline_hps = torch.FloatTensor(pipeline_hps).to(self.device)
        ce = ce.to(self.device)
        acc = acc.to(self.device)

        if self.return_acc:
            metric = acc
        else:
            metric = ce

        if return_context:
            context = self.get_context(
                dataset, mode=mode, max_context_size=self.max_context_size
            )
            pipeline_hps_context, metric_context, acc_per_pipeline_context = context

            with torch.no_grad():
                mean = acc_per_pipeline_context.mean()
                std = acc_per_pipeline_context.std()

                if torch.isnan(std) or std == 0:
                    std = 1
                metric_context = (metric_context - mean) / std
                metric = (metric - mean) / std
                acc_per_pipeline = (acc_per_pipeline - mean) / std

            context = torch.cat(
                [pipeline_hps_context.squeeze(1), metric_context.unsqueeze(-1)], axis=-1
            )
            context = context.unsqueeze(0).repeat(batch_size, 1, 1)

            return pipeline_hps, metric, context, acc_per_pipeline

        return pipeline_hps, metric, acc_per_pipeline


class SetTransformer(nn.Module):
    def __init__(self, dim_in, hidden_dim=64, num_heads=4, num_seeds=1, out_dim=1):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=dim_in, dim_out=hidden_dim, num_heads=num_heads),
            SAB(dim_in=hidden_dim, dim_out=hidden_dim, num_heads=num_heads),
        )
        self.dec = nn.Sequential(
            PMA(dim=hidden_dim, num_heads=hidden_dim, num_seeds=num_seeds),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        x = nn.Sigmoid()(x)
        return x.squeeze(-1)


class EnsembleNet(nn.Module):
    def __init__(
        self,
        dim_in,
        hidden_dim=64,
        hidden_dim_ff=32,
        num_heads=4,
        num_seeds=1,
        out_dim=32,
        out_dim_ff=1,
        return_context=False,
    ):
        super().__init__()
        # self.encoder = SetTransformer(dim_in, hidden_dim, num_heads, num_seeds, out_dim)
        self.encoder = SetTransformer(
            dim_in + 1, hidden_dim, num_heads, num_seeds, out_dim
        )

        if return_context:
            self.hidden_layer = nn.Linear(
                in_features=out_dim * 2, out_features=hidden_dim_ff
            )
        else:
            self.hidden_layer = nn.Linear(in_features=out_dim, out_features=hidden_dim_ff)
        self.out_layer = nn.Linear(in_features=hidden_dim_ff, out_features=out_dim_ff)
        self.return_context = return_context

    def forward(self, x, context=None, y=None):
        if y is None:
            y = torch.zeros(x.shape[0], x.shape[1], 1).to(x.device)
        else:
            ones_pct = 1 - 1 / x.shape[1]
            y = y.unsqueeze(-1)
            mask = torch.bernoulli(torch.full(y.shape, ones_pct)).to(x.device)
            y = y.to(x.device)
            y = y * mask

        if context is None and self.return_context:
            raise ValueError("return_context is True but no context was provided")

        x = torch.cat([x, y], axis=-1)
        x = self.encoder(x)

        if self.return_context:
            context = self.encoder(context)
            x = torch.cat([x, context], axis=-1)
        x = self.hidden_layer(x)
        x = nn.ReLU()(x)
        x = self.out_layer(x)
        # x = nn.Sigmoid()(x)
        x = nn.LeakyReLU()(x)
        return x


def train(
    model,
    loader,
    return_context=False,
    val_freq=50,
    num_epochs=5000,
    lr=0.0001,
    clip_value=1.0,
    criterion_type="listwise",
):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.L1Loss().cuda()
    criterion = RankLoss(criterion_type)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for i in range(num_epochs):
        losses = []

        if return_context:
            x, y, context, acc_per_pipeline = loader.get_batch(return_context=True)
            y_pred = model(x, context, acc_per_pipeline)
        else:
            x, y, acc_per_pipeline = loader.get_batch(return_context=False)
            y_pred = model(x, None, acc_per_pipeline)

        # x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda()
        # y_pred = model(x)
        loss = criterion(y_pred.reshape(y.shape), y)

        if torch.isnan(y_pred).any() or torch.isnan(loss):
            print("nan")
            continue
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        losses.append(loss.item())
        # print(loss.item())

        if i % val_freq == 0:
            val_loss = test(
                model, loader, criterion, mode="val", return_context=return_context
            )
            print("val loss", val_loss)
            print("train loss", np.median(losses))
            scheduler.step()
            losses = []

    return losses


def test(model, loader, criterion, mode="val", return_context=False, num_epochs=200):
    losses = []
    # criterion = nn.L1Loss().cuda()

    for _ in range(num_epochs):
        with torch.no_grad():
            if return_context:
                x, y, context, acc_per_pipeline = loader.get_batch(
                    mode=mode, return_context=True
                )
                y_pred = model(x, context, acc_per_pipeline)
            else:
                x, y, acc_per_pipeline = loader.get_batch(mode=mode, return_context=False)
                y_pred = model(x, acc_per_pipeline)

            loss = criterion(y_pred.reshape(y.shape), y)
            losses.append(loss.item())

    return np.median(losses)


if __name__ == "__main__":
    return_context = True
    criterion_type = "weighted_listwise"
    loader = EnsembleLoader(
        batch_size=8,
        max_num_pipelines=10,
        return_context=return_context,
        max_context_size=30,
    )
    x = loader.get_batch()

    model = SetTransformer(dim_in=x[0].shape[-1], hidden_dim=128)
    model = EnsembleNet(
        dim_in=x[0].shape[-1],
        hidden_dim=512,
        hidden_dim_ff=128,
        return_context=return_context,
    )
    losses = train(
        model,
        loader,
        return_context=return_context,
        val_freq=200,
        num_epochs=100000,
        criterion_type=criterion_type,
    )
    print("Done")
