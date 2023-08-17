# type: ignore
# pylint: skip-file

import numpy as np
import torch
from rank_loss import RankLoss
from scipy.stats import kendalltau
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(
    model,
    loader,
    val_freq=50,
    num_epochs=5000,
    lr=0.0001,
    clip_value=1.0,
    criterion_type="listwise",
    num_epochs_test=100,
    observed_pipeline_ids=None,
    model_path=None,
):
    best_score = -np.inf
    model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.L1Loss().cuda()
    criterion = RankLoss(criterion_type)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    losses = []
    for i in range(num_epochs):
        x, y, y_per_pipeline = loader.get_batch(
            num_encoders=model.num_encoders, observed_pipeline_ids=observed_pipeline_ids
        )
        y_pred = model(x, y_per_pipeline)
        loss = torch.Tensor([0]).cuda()
        for y_pred_, y_ in zip(y_pred, y):
            loss += (criterion(y_pred_.reshape(y_.shape), y_)) / model.num_encoders

        if torch.isnan(loss):
            print("nan")
            continue

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        losses.append(loss.item())
        # print(loss.item())

        if i % val_freq == 0 and observed_pipeline_ids is None:
            val_loss, k_coef = test(
                model, loader, criterion, mode="val", num_epochs=num_epochs_test
            )
            ece = ECE(model, loader)
            print("val loss", val_loss)
            print("train loss", np.median(losses))
            print("ECE", ece)
            print("kendall", k_coef)

            if k_coef > best_score and model_path is not None:
                best_score = k_coef
                torch.save(model.state_dict(), model_path)
                print("Saving model...")

            scheduler.step()
    model.eval()
    return losses


def test(model, loader, criterion, mode="val", num_epochs=100, same_input=False):
    losses = []
    # criterion = nn.L1Loss().cuda()
    k = 0
    model.eval()
    for _ in range(num_epochs):
        with torch.no_grad():
            x, y, y_per_pipeline = loader.get_batch(
                num_encoders=model.num_encoders, same_input=same_input, mode=mode
            )
            y_pred = model(x, y_per_pipeline)

            loss = 0
            for y_pred_, y_ in zip(y_pred, y):
                loss += (criterion(y_pred_.reshape(y_.shape), y_)) / model.num_encoders
            losses.append(loss.item())
            pred = model.predict(x, y_per_pipeline)
            k += kendalltau(pred[0].cpu().numpy(), -y[0].cpu().numpy())[0]

    return np.median(losses), k / num_epochs


def ECE(model, loader, num_bins=50):
    # https://www.eng.biu.ac.il/goldbej/files/2023/04/LIor_ISBI_2023.pdf
    with torch.no_grad():
        x, y, y_per_pipeline = loader.get_batch(
            num_encoders=model.num_encoders, batch_size=1000
        )
        true_rank = model.get_rank(-y[0]).cpu().numpy()
        mean, std = model.predict(x, y_per_pipeline)
        mean = mean.cpu().numpy()
        var = torch.pow(std, 2).cpu().numpy()
        hist, bin_edges = np.histogram(var, bins=num_bins)
        bin_indices = np.digitize(var, bin_edges)

        rmv = np.zeros(num_bins)
        rmse = np.zeros(num_bins)
        ece = 0
        for i in range(1, num_bins):
            if np.sum(bin_indices == i) > 0:
                rmv[i - 1] = np.sqrt(np.mean(var[bin_indices == i]))
                rmse[i - 1] = np.sqrt(
                    np.mean(
                        np.power(true_rank[bin_indices == i] - mean[bin_indices == i], 2)
                    )
                )

                if rmv[i - 1] == 0:
                    continue
                ece += np.abs(rmv[i - 1] - rmse[i - 1]) / rmv[i - 1]
        ece = ece / num_bins

        if np.isnan(ece):
            print("ece is nan")

        return ece
