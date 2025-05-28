import numpy as np
import torch
from torch import nn


class RankLoss(nn.Module):
    def __init__(self, type="pointwise"):
        super().__init__()

        if type == "pointwise":
            self.loss = self.pointwise_loss
        elif type == "pairwise":
            self.loss = self.pairwise_loss
        elif type == "listwise":
            self.loss = self.listwise_loss
        elif type == "weighted_listwise":
            self.loss = self.weighted_listwise_loss
        else:
            raise ValueError("Loss type not supported")

    def pairwise_loss(self, predictions, targets):
        loss = 0
        count = 0
        targets = -targets
        for i in range(len(predictions)):
            for j in range(len(predictions)):
                if targets[i] > targets[j]:
                    loss += torch.log(nn.Sigmoid()(predictions[i] - predictions[j]))
                    count += 1
        return loss / count

    def pointwise_loss(self, predictions, targets):
        # return nn.MSELoss()(predictions, -targets)
        return nn.L1Loss()(predictions, targets)

    def listwise_loss(self, predictions, targets):
        return -self.listMLE(predictions.reshape(1, -1), -targets.reshape(1, -1))

    def weighted_listwise_loss(self, predictions, targets):
        return -self.listMLE_weighted(predictions.reshape(1, -1), -targets.reshape(1, -1))

    def listMLE(self, y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        # shuffle for randomised tie resolution
        if len(y_pred.shape) < 2:
            y_pred = y_pred.unsqueeze(0)
        if len(y_true.shape) < 2:
            y_true = y_true.unsqueeze(0)

        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == padded_value_indicator

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(
            preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
        ).flip(dims=[1])

        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        return torch.mean(torch.sum(observation_loss, dim=1))

    def listMLE_weighted(self, y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == padded_value_indicator

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(
            preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
        ).flip(dims=[1])

        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        ####### Weighting extension
        # Weighted ranking because it is more important to get the the first ranks right than the rest.
        weight = np.log(
            np.arange(observation_loss.shape[-1]) + 2
        )  # Adding 2 to prevent using log(0) & log(1) as weights.
        weight = np.array(weight, dtype=np.float32)
        weight = torch.from_numpy(weight)[None, :].to(observation_loss.device)
        observation_loss = observation_loss / weight
        #######

        return torch.mean(torch.sum(observation_loss, dim=1))

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)
