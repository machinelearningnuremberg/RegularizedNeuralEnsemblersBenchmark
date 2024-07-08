from __future__ import annotations

import os
from pathlib import Path

from scipy import stats
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from typing_extensions import Literal
from sklearn.model_selection import KFold

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..samplers.random_sampler import RandomSampler
from .base_ensembler import BaseEnsembler

try:
    import wandb
    WAND_AVAILABLE = True
except: 
    WAND_AVAILABLE = False


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim):
        super(FeedforwardNetwork, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NeuralEnsembler(BaseEnsembler):
    """Neural (End-to-End) Ensembler."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cuda"),
        prediction_device: torch.device = torch.device("cpu"),
        epochs: int = 1000,
        use_wandb: bool = False,
        checkpoint_freq: int = 1000,
        run_name: str = None,
        num_pipelines: int = 64,
        project_name: str = "Training_NE",
        ne_learning_rate: float =0.0001,
        ne_hidden_dim: int = 512,
        ne_reg_term_div: float = 0.0,
        ne_add_y: bool = True,
        ne_num_layers: int = 2,
        ne_reg_term_norm: float = 0.0,
        ne_net_type: str = "ffn",
        ne_num_heads: int = 4,
        ne_mode: Literal["inference", "pretraining"] = "inference",
        ne_checkpoint_name: str = "auto",
        ne_resume_from_checkpoint: bool = False,
        ne_use_mask: bool = True,
        ne_unique_weights_per_function: bool = False,
        ne_dropout_rate: float = 0.,
        ne_batch_size: int = 2048,
        ne_auto_dropout: bool = False,
        ne_weight_thd: float = 0.,
        ne_dropout_dist: str | None = None,
        ne_omit_output_mask: bool = True,
        **kwargs
    ) -> None:
        super().__init__(metadataset=metadataset, device=device)

        self.hidden_dim = ne_hidden_dim
        self.reg_term_div = ne_reg_term_div
        self.device = device
        self.add_y = ne_add_y
        self.num_layers = ne_num_layers
        self.reg_term_norm = ne_reg_term_norm
        self.net_type = ne_net_type
        self.prediction_device = prediction_device
        self.mode = ne_mode
        self.ne_batch_size = ne_batch_size
        self.learning_rate = ne_learning_rate
        self.epochs = epochs
        self.use_wandb = use_wandb
        self.num_heads = ne_num_heads
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_name = ne_checkpoint_name
        self.resume_from_checkpoint = ne_resume_from_checkpoint
        self.use_mask = ne_use_mask
        self.unique_weights_per_function = ne_unique_weights_per_function
        self.run_name = run_name
        self.num_pipelines = num_pipelines #used for pretraining
        self.dropout_rate = ne_dropout_rate
        self.auto_dropout = ne_auto_dropout
        self.project_name = project_name # for wandb
        self.weight_thd = ne_weight_thd
        self.dropout_dist = ne_dropout_dist
        self.omit_output_mask = ne_omit_output_mask
        self.training = True
        self.predefined_pipeline_ids = None
        self.y_max = 1

        if self.metadataset.metric_name == "relative_absolute_error":
            self.criterion = nn.L1Loss()
            self.task_type = "regression"
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.task_type = "classification"
   
        if self.use_wandb and self.mode == "pretraining" and WAND_AVAILABLE:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                settings=wandb.Settings(start_method="fork"),

            )

    def set_state(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cuda"),
        mode: str = "inference",
    ) -> None:
        self.metadataset = metadataset
        self.device = device
        self.mode = mode

    def batched_prediction(
        self, X, base_functions
    ):

        _, num_samples, num_classes, num_pipelines = X.shape
        idx = np.arange(num_samples)
        # np.random.shuffle(idx)
        outputs = []
        weights = []

        if self.task_type == "regression":
            #base_functions is not necessary to transform because they permit to mantain the original scale
            X /= self.y_max

        for i in range(0, num_samples, self.ne_batch_size):
            range_idx = idx[range(i, min(i + self.ne_batch_size, num_samples))]

            with torch.no_grad():
                output, w = self.net(
                    x=X[:, range_idx],
                    base_functions=base_functions[:, range_idx]
                )
            w = w.transpose(2, 3).transpose(
                1, 2
            )  # Expected shape [BATCH SIZE X NUM_PIPELINES X NUM_SAMPLES X NUM_CLASSES]
            outputs.append(output.to(self.prediction_device))
            weights.append(w.to(self.prediction_device))

        return torch.cat(outputs, axis=-2).to(self.prediction_device), torch.cat(
            weights, axis=-2
        ).to(self.prediction_device)

    def get_auto_weight_thd(self):
        return 1 / (self.metadataset.get_num_classes() * self.metadataset.get_num_pipelines())

    def get_weights(self, X_obs, X_context=None, y_context=None):
        base_functions = (
            self.metadataset.get_predictions([X_obs])[0]
            .transpose(0, 1)
            .transpose(2, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        _, weights = self.batched_prediction(
            X=base_functions,
            base_functions=base_functions
        )
        weights = self.prune_weights(weights)
        return weights

    def prune_weights(self, weights):

        if self.weight_thd == -1 :
            weight_thd = self.get_auto_weight_thd()
        else:
            weight_thd = self.weight_thd
        
        weight_pruner = nn.Threshold(weight_thd, 0.)
        weights = weight_pruner(weights)
        weights_sum = weights.sum(-3, keepdim=True).sum(-1, keepdim=True)
        weights /= weights_sum
        return weights

    def sample(self, X_obs, **kwargs) -> tuple[list, float]:
        """Fit neural ensembler, output ensemble WITH weights"""
        
        self.X_obs = X_obs
        best_ensemble = None
        weights = None
        # this has to change when using more batches
        base_functions = (
            self.metadataset.get_predictions([X_obs])[0]
            .transpose(0, 1)
            .transpose(2, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        ##this has to change when using more batches
        y = self.metadataset.get_targets().unsqueeze(0).to(self.device)
        
        if self.auto_dropout:
            self.net = self.auto_dropout_and_fit(
                X_train=base_functions, y_train=y, base_functions_train=base_functions
            )           
        else:
            self.net = self.fit_net(
                X_train=base_functions, y_train=y, base_functions_train=base_functions
            )
        _, weights = self.batched_prediction(
            X=base_functions, base_functions=base_functions
        )
        best_ensemble = X_obs
        _, best_metric, _, _ = self.metadataset.evaluate_ensembles_with_weights(
            [best_ensemble], weights
        )
        self.best_ensemble = best_ensemble
        return best_ensemble, best_metric

    def send_to_device(self, *args):
        output = []
        for arg in args:
            if arg is not None:
                output.append(arg.to(self.device))
            else:
                output.append(arg)
        return output

    def get_batch(self, X_train, base_functions_train, y_train):
        _, num_samples, num_classes, num_base_functions = X_train.shape
    
        idx = np.random.randint(0, min(self.ne_batch_size, num_samples), self.ne_batch_size)
       # return (X_train[:, idx], base_functions_train[:, idx], y_train[:, idx])
        return (X_train, base_functions_train, y_train)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def auto_dropout_and_fit(self,
                            X_train, 
                            y_train,
                            base_functions_train,
                            num_folds=3,
                            dropout_rate_list = [0, 0.25, 0.5, 0.75]
                            ):
        
        _, num_samples, num_classes, num_base_functions = X_train.shape
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        kf = KFold(n_splits=num_folds)        
        val_loss_list = []
        for dropout_rate in dropout_rate_list:
            val_loss = 0
            for train_idx, val_idx in kf.split(idx):
                net = self.fit_net(X_train[:,train_idx], 
                                y_train[:,train_idx], 
                                base_functions_train[:,train_idx],
                                dropout_rate
                                )
                temp_val_loss = self.validate(net, X_train[:,val_idx], 
                                        y_train[:,val_idx],
                                        base_functions_train[:,val_idx])
                val_loss += temp_val_loss / len(dropout_rate_list)
            val_loss_list.append(val_loss)
        
        best_dropout_rate = dropout_rate_list[np.argmin(val_loss_list)]
        return self.fit_net(X_train, y_train, base_functions_train, best_dropout_rate)

    def fit_net(
        self,
        X_train,
        y_train,
        base_functions_train,
        dropout_rate: float | None = None
    ):
        
        self.training = True
        if self.net_type == "ffn":
            NetClass = EFFNet
            input_dim = X_train.shape[-1]
            output_dim = base_functions_train.shape[-1]  # [NUMBER OF BASE FUNCTIONS]
        elif self.net_type == "simple":
            NetClass = ENetSimple
            input_dim = 1
            output_dim = base_functions_train.shape[-1]*base_functions_train.shape[-2]
        else:
            raise NotImplementedError()
        
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        model = NetClass(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            add_y=self.add_y,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            unique_weight_per_base_function=self.unique_weights_per_function,
            dropout_rate=dropout_rate,
            dropout_dist=self.dropout_dist,
            omit_output_mask=self.omit_output_mask
        )

        if self.task_type == "regression":
            self.y_max = base_functions_train.max()
            y_train /= self.y_max
            base_functions_train /= self.y_max
            X_train /= self.y_max

        elif self.task_type == "classification":
            y_train = torch.tensor(y_train, dtype=torch.long)

        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        if self.resume_from_checkpoint:
            self.load_checkpoint(model)

        X_train, y_train, base_functions_train, model = self.send_to_device(
            X_train, y_train, base_functions_train, model
        )

        model.train()
        for epoch in range(self.epochs):
            batch_data = self.get_batch(X_train, base_functions_train, y_train)
            loss, w = self.fit_one_epoch(model, optimizer, batch_data)
            print("Epoch", epoch, "Loss", loss.item(), "W:", w.median().item())

        model.eval()
        self.training = False
        return model

    def validate(self, model, X_val, y_val, base_functions_val):

        X_val, y_val, base_functions_val = self.send_to_device(X_val, y_val, base_functions_val)
        output, w = model(X_val, base_functions_val)
        logits = self.metadataset.get_logits_from_probabilities(output)
        _, num_samples, num_classes, num_base_functions = X_val.shape

        val_loss = self.criterion(logits.reshape(-1, num_classes), y_val.reshape(-1)).item()

        return val_loss

    def fit_one_epoch(self, model, optimizer, batch_data):
        optimizer.zero_grad()
        X_batch, base_functions_batch, y_batch = batch_data[:3]
        _, num_samples, num_classes, num_base_functions = X_batch.shape
        output, w = model(X_batch, base_functions_batch)

        if self.task_type == "classification":
            logits = self.metadataset.get_logits_from_probabilities(output)
            loss = self.criterion(logits.reshape(-1, num_classes), y_batch.reshape(-1))

        else: #assuming task is regression
            loss = self.criterion(output.reshape(-1), y_batch.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        return loss, w

    def save_checkpoint(self, model,
                        optimizer: dict = None,
                        epoch: int = 0,
                        loss: float = 0.0):

        if self.checkpoint_name == "auto":
            test_id =  self.metadataset.meta_split_ids[-1][0]
            checkpoint_name = f"neural_ensembler_{test_id}.pt"
        else:
            checkpoint_name = self.checkpoint_name

        if optimizer is not None:
            optimizer_state_dict = optimizer.state_dict()
        complete_path = Path(os.path.abspath(__file__)).parent.parent.parent / "checkpoints" / checkpoint_name
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_state_dict,
                    'loss': loss,
                    }, complete_path)

    def load_checkpoint(self, model,
                        optimizer: dict = None):

        if self.checkpoint_name == "auto":
            test_id =  self.metadataset.meta_split_ids[-1][0]
            checkpoint_name = f"neural_ensembler_{test_id}.pt"
        else:
            checkpoint_name = self.checkpoint_name

        complete_path = Path(os.path.abspath(__file__)).parent.parent.parent / "checkpoints" / checkpoint_name
        checkpoint = torch.load(complete_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return model, optimizer, epoch, loss




class ENetSimple(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        output_dim=1,
        num_layers=2,
        dropout_rate=0.,
        num_heads=1,
        add_y=False,
        mask_prob=0.5,
        **kwargs,
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.mask_prob = mask_prob
        self.dropout_rate = dropout_rate

        custom_weights_fc1 = torch.randn(
            output_dim
        )  # Custom weights for the first fully connected layer
        self.weight = nn.Parameter(custom_weights_fc1)

    def forward(
        self, x, base_functions
    ):

        batch_size, num_samples, num_classes, num_base_functions = x.shape
        mask = None
        if self.dropout_rate > 0 and self.training:
            mask= (torch.rand(size=(num_base_functions,)) > self.dropout_rate).float().to(x.device)
            for i, dim in enumerate([batch_size, num_samples, num_classes]):
                mask = torch.repeat_interleave(
                    mask.unsqueeze(i), dim, dim=i
                )
            x = (x*mask)/(1-self.dropout_rate)
            base_functions = base_functions*mask

        w = torch.repeat_interleave(
            self.weight.reshape(1, -1), base_functions.shape[1], dim=0
        )

        w = w.reshape(batch_size, num_samples, -1)
        
        if mask is not None:
            mask = mask.reshape(batch_size, num_samples, -1)
            w = w.masked_fill(mask == 0, -1e9)
            
        w_norm = torch.nn.functional.softmax(w, dim=-1)
        w_norm = w_norm.reshape(
            batch_size, num_samples, num_classes, num_base_functions
        )
        x = torch.multiply(base_functions, w_norm).sum(axis=-1)

        return x, w_norm


class EFFNet(nn.Module):  # Sample as Sequence
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        output_dim=1,
        num_layers=2,
        dropout_rate=0,
        num_heads=1,
        add_y=False,
        mask_prob=0.5,
        inner_batch_size=50,
        unique_weight_per_base_function=False,
        dropout_dist=None,
        omit_output_mask=False
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.mask_prob = mask_prob
        self.inner_batch_size = inner_batch_size
        self.unique_weight_per_base_function = unique_weight_per_base_function
        self.dropout_dist = dropout_dist
        self.omit_output_mask = omit_output_mask

        if unique_weight_per_base_function:
            num_layers=-1
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        first_encoder_modules = [
            nn.Sequential( nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ]
        first_encoder_modules.append(nn.ReLU())

        self.first_encoder = nn.ModuleList(first_encoder_modules)
        self.second_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.ReLU())

        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate
        self.dropout_is_active = (dropout_rate > 0.) or (dropout_dist != None)


    def get_batched_weights(self, x, base_functions):

        batch_size, num_samples, num_classes, num_base_functions = x.shape
        x = x.transpose(1, 2).reshape(-1, num_samples, num_base_functions)

        # [(BATCH SIZE X NUMBER OF CLASSES) X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS]
        x = self.embedding(x)
        for encoder in self.first_encoder:
            x = encoder(x)

        x = x.transpose(0, 1).unsqueeze(0)  # use batch size here

        return x

    def sample_dropout_rate(self):
        if self.dropout_dist is None:
            dropout_rate = self.dropout_rate
        elif self.dropout_dist == "uniform":
            dropout_rate = np.random.uniform()
        elif self.dropout_dist == "truncated_normal":
            loc = self.dropout_rate
            scale = 0.1
            quantile1 = stats.norm.cdf(0, loc=loc, scale=scale)
            quantile2 = stats.norm.cdf(1, loc=loc, scale=scale)
            dropout_rate = stats.norm.ppf(
                    np.random.uniform(quantile1, quantile2, size=1),
                    loc=loc,
                    scale=scale,
                ).item()       
        else:
            raise ValueError("Dropout dist is not implemented.")
        return dropout_rate

    def get_mask_and_scaling_factor(self, num_base_functions, device):
        dropout_rate = self.sample_dropout_rate()
        mask= (torch.rand(size=(num_base_functions,)) > dropout_rate).float().to(device)
        scaling_factor = 1./(1.- dropout_rate)
        return mask, scaling_factor

    def forward(
        self, x, base_functions
    ):
        # X = [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        batch_size, num_samples, num_classes, num_base_functions = x.shape
        #num_query_samples = x.shape[1]

        if self.dropout_is_active and self.training:
            mask, scaling_factor = self.get_mask_and_scaling_factor(num_base_functions, x.device)
            for i, dim in enumerate([batch_size, num_samples, num_classes]):
                mask = torch.repeat_interleave(
                    mask.unsqueeze(i), dim, dim=i
                )
            x = (x*mask)*scaling_factor
            base_functions = base_functions*mask
        else:
            mask = None

        w = []
        idx = np.arange(num_classes)
        for i in range(0, num_classes, self.inner_batch_size):
            range_idx = idx[range(i, min(i + self.inner_batch_size, num_classes))]
            temp_w = self.get_batched_weights(x = x[:,:,range_idx],
                                              base_functions = base_functions[:,:,range_idx])
            w.append(temp_w)

        w = torch.cat(w, axis=2)

        if self.unique_weight_per_base_function:
            w = w.mean(axis=2)
            w = self.second_encoder(w)
            w = torch.repeat_interleave(
                w.unsqueeze(2), num_classes, dim=2
            )
            
        w = self.out_layer(w)

        #num_classes changed
        #batch_size, num_samples, num_classes, num_base_functions = w.shape
        
        if (mask is not None) and (not self.omit_output_mask):
            w = w.masked_fill(mask == 0, -1e9)

        w = w.reshape(batch_size, num_samples, -1)
        w_norm = torch.nn.functional.softmax(w, dim=-1)
        
        w_norm = w_norm.reshape(
            batch_size, num_samples, num_classes, num_base_functions
        )

        w_norm = torch.divide(w_norm, torch.FloatTensor([1/num_classes]).to(w_norm.device))
        x = torch.multiply(base_functions, w_norm).sum(axis=-1)
        # x.shape: [BATCH_SIZE, NUM_SAMPLES, NUM_CLASSES]
        # w_norm.shape : [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        return x, w_norm
