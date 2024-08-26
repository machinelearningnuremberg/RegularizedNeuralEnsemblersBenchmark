from __future__ import annotations

import os
from pathlib import Path

from scipy import stats
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from typing_extensions import Literal
from sklearn.model_selection import KFold
import copy

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..samplers.random_sampler import RandomSampler
from .base_ensembler import BaseEnsembler

try:
    import wandb
    WAND_AVAILABLE = True
except: 
    WAND_AVAILABLE = False

class Dataset:
    def __init__(self, X, y=None, batch_size=2048):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = len(X)
    
    def __iter__(self):
        for i in range(self.num_samples):
            down = i*self.batch_size
            up = min((i+1)*self.batch_size, self.num_samples)
            if self.y is not None:
                yield self.X[down:up], self.y[down:up]
            else:
                yield self.X[down:up]

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
        metadataset: BaseMetaDataset | None = None,
        device: torch.device = torch.device("cuda"),
        prediction_device: torch.device = torch.device("cpu"),
        normalize_performance: bool = False,
        use_wandb: bool = False,
        checkpoint_freq: int = 1000,
        run_name: str = None,
        num_pipelines: int = 64,
        project_name: str = "Training_NE",
        metric_name: str = "error",
        ne_learning_rate: float =0.0001,
        ne_hidden_dim: int = 32,
        ne_reg_term_div: float = 0.0,
        ne_add_y: bool = True,
        ne_num_layers: int = 3,
        ne_reg_term_norm: float = 0.0,
        ne_net_type: str = "ffn",
        ne_num_heads: int = 4,
        ne_mode: Literal["inference", "pretraining"] = "inference",
        ne_checkpoint_name: str = "auto",
        ne_resume_from_checkpoint: bool = False,
        ne_use_mask: bool = True,
        ne_dropout_rate: float = 0.,
        ne_batch_size: int = 2048,
        ne_auto_dropout: bool = False,
        ne_weight_thd: float = 0.,
        ne_dropout_dist: str | None = None,
        ne_omit_output_mask: bool = True,
        ne_net_mode: str = "combined",
        ne_epochs: int = 1000,
        ne_window_average: int = 100,
        ne_patience: int = -1,
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
        self.use_wandb = use_wandb
        self.num_heads = ne_num_heads
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_name = ne_checkpoint_name
        self.resume_from_checkpoint = ne_resume_from_checkpoint
        self.use_mask = ne_use_mask
        self.run_name = run_name
        self.num_pipelines = num_pipelines #used for pretraining
        self.dropout_rate = ne_dropout_rate
        self.auto_dropout = ne_auto_dropout
        self.project_name = project_name # for wandb
        self.weight_thd = ne_weight_thd
        self.dropout_dist = ne_dropout_dist
        self.omit_output_mask = ne_omit_output_mask
        self.net_mode = ne_net_mode
        self.normalize_performance = normalize_performance
        self.epochs = ne_epochs
        self.window_average = ne_window_average
        self.patience = ne_patience
        
        self.training = True
        self.predefined_pipeline_ids = None
        self.y_scale = 1
        self.class_prob = torch.FloatTensor([1,1,1,1])
        self.best_ensemble = []

        if self.metadataset is not None:
            self.metric_name = metadataset.metric_name
        else:
            #dummy metadataset
            self.metadataset = BaseMetaDataset("")
            self.metric_name = metric_name

        if self.metric_name == "absolute_relative_error":
            self.criterion = nn.L1Loss()
            self.task_type = "regression"
        elif self.metric_name == "mse":
            self.criterion = nn.MSELoss()
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
        self, X
    ):

        num_samples, num_classes, num_pipelines = X.shape
        idx = np.arange(num_samples)
        # np.random.shuffle(idx)
        outputs = []
        weights = []

        if self.task_type == "regression":
            X /= self.y_scale.to(X.device)

        for i in range(0, num_samples, self.ne_batch_size):
            range_idx = idx[range(i, min(i + self.ne_batch_size, num_samples))]

            with torch.no_grad():
                output, w = self.net(
                    x=X[range_idx]
                )
            outputs.append(output.to(self.prediction_device))

            if w is not None:
                w = w.transpose(1, 2).transpose(
                    0, 1
                )  # Expected shape [NUM_PIPELINES X NUM_SAMPLES X NUM_CLASSES]

                weights.append(w.to(self.prediction_device))

        outputs = torch.cat(outputs, axis=-2).to(self.prediction_device)
        weights = torch.cat(
            weights, axis=-2
        ).to(self.prediction_device) if len(weights)>0 else None

        if self.task_type == "regression":
            outputs *= self.y_scale.to(outputs.device)

        return outputs, weights

    def get_auto_weight_thd(self):
        return 1 / (self.metadataset.get_num_classes() * self.metadataset.get_num_pipelines())

    def get_weights(self, X_obs):
        base_functions = (
            self.metadataset.get_predictions([X_obs])[0]
            .transpose(0, 1)
            .transpose(2, 1)
            .to(self.device)
        )

        _, weights = self.batched_prediction(
            X=base_functions,
        )
        if weights is not None:
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

        base_functions = (
            self.metadataset.get_predictions([X_obs])[0]
            .transpose(0, 1)
            .transpose(2, 1)
            .to(self.device)
        )
        y = self.metadataset.get_targets().to(self.device)
        
        if self.auto_dropout:
            self.net = self.auto_dropout_and_fit(
                X_train=base_functions, y_train=y
            )           
        else:
            self.net = self.fit_net(
                X_train=base_functions, y_train=y
            )
        self.best_ensemble = X_obs

        best_metric = self.evaluate_on_split(split="valid")
        return best_ensemble, best_metric

    def send_to_device(self, *args):
        output = []
        for arg in args:
            if arg is not None:
                output.append(arg.to(self.device))
            else:
                output.append(arg)
        return output

    def get_batch(self, X_train, y_train):
        num_samples, num_classes, num_base_functions = X_train.shape
    
        idx = np.random.randint(0, num_samples, self.ne_batch_size)
        return (X_train[idx], y_train[idx])
        #return (X_train, y_train)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_on_split(
        self,
        split: str = "test",
    ):

        self.metadataset.set_state(dataset_name=self.metadataset.dataset_name,
                                        split = split)
        
        y = self.metadataset.get_targets().to(self.device)
        y_pred = self.get_y_pred()
        metric = self.metadataset.score_y_pred(y_pred, y)
        
        if self.normalize_performance:
            metric = self.metadataset.normalize_performance(metric)
        
        return metric

    def get_y_pred(self):
        
        base_functions = (
            self.metadataset.get_predictions([self.best_ensemble])[0]
            .transpose(0, 1)
            .transpose(2, 1)
            .to(self.device)
        )

        y_pred = self.batched_prediction(
            X=base_functions,
        )[0]

        return y_pred
    
    def from_list_to_tensor(self, X):
        X_concat = []
        for X_temp in X:
            X_concat.append(torch.FloatTensor(X_temp).unsqueeze(-1))
        return torch.cat(X_concat, axis=-1).to(self.device)
    
    def fit(self, X, y):
        y = torch.tensor(y)
        X = self.from_list_to_tensor(X)

        if self.task_type == "regression":
            y = y.float()
        self.net = self.fit_net(X,y)

    def predict(self, X):
        X = self.from_list_to_tensor(X)
        y_pred = self.batched_prediction(
            X=X,
        )[0]

        return y_pred
          
    def auto_dropout_and_fit(self,
                        X_train, 
                        y_train,
                        num_folds=3,
                        dropout_rate_list = [0, 0.25, 0.5, 0.75]
                        ):
    
        num_samples, num_classes, num_base_functions = X_train.shape
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        kf = KFold(n_splits=num_folds)        
        val_loss_list = []
        for dropout_rate in dropout_rate_list:
            val_loss = 0
            for train_idx, val_idx in kf.split(idx):
                net = self.fit_net(X_train[train_idx], 
                                y_train[train_idx], 
                                dropout_rate
                                )
                temp_val_loss = self.validate(net, X_train[val_idx], 
                                        y_train[:,val_idx])
                val_loss += temp_val_loss / len(dropout_rate_list)
            val_loss_list.append(val_loss)
        
        best_dropout_rate = dropout_rate_list[np.argmin(val_loss_list)]
        return self.fit_net(X_train, y_train, best_dropout_rate)

    def start_loss_tracker(self): 
        self.loss_history = []
        self.loss_window_avg = self.lowest_loss = np.inf
        self.patience_counter = 0

    def update_loss_tracker(self, loss, epoch):
        loss = loss.item()
        self.loss_history.append(loss)
        loss_window_avg = np.mean(self.loss_history[-self.window_average:])
        if loss_window_avg < self.lowest_loss:
            self.lowest_loss = loss_window_avg

        if epoch > self.window_average and self.patience >= 0:
            if loss_window_avg > self.lowest_loss:
                self.patience_counter += 1
            else:
                self.patience_counter = 0 

        if self.patience_counter == self.patience:
            stop = True
        else:
            stop = False
        
        return stop, loss_window_avg
    
    def fit_net(
        self,
        X_train,
        y_train,
        dropout_rate: float | None = None
    ):
        
        self.training = True
        if self.net_type == "ffn":
            NetClass = EFFNet
            input_dim = X_train.shape[-1]
            output_dim = X_train.shape[-1]  # [NUMBER OF BASE FUNCTIONS]
        elif self.net_type == "simple":
            NetClass = ENetSimple
            input_dim = 1
            output_dim = X_train.shape[-1]*X_train.shape[-2]
        else:
            raise NotImplementedError()
        
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        num_classes = X_train.shape[-2]

        model = NetClass(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            add_y=self.add_y,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=dropout_rate,
            dropout_dist=self.dropout_dist,
            omit_output_mask=self.omit_output_mask,
            mode=self.net_mode,
            task_type=self.task_type,
            num_classes=num_classes
        )

        if self.task_type == "regression":
            self.y_scale = X_train.mean()
            y_train /= self.y_scale
            X_train /= self.y_scale

        elif self.task_type == "classification":
            y_train = torch.tensor(y_train, dtype=torch.long)

        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        if self.resume_from_checkpoint:
            self.load_checkpoint(model)

        X_train, y_train, model = self.send_to_device(
            X_train, y_train, model
        )

        self.start_loss_tracker()
        model.train()
        for epoch in range(self.epochs):
            batch_data = self.get_batch(X_train, y_train)
            loss, w = self.fit_one_epoch(model, optimizer, batch_data)
            stop, loss_window_avg = self.update_loss_tracker(loss, epoch)
            print("Epoch", epoch, "Loss", loss.item(), "Loss-Window-Avg", loss_window_avg)

            if stop: break

        model.eval()
        self.training = False
        return model

    def validate(self, model, X_val, y_val):

        X_val, y_val = self.send_to_device(X_val, y_val)
        output, w = model(X_val)
        logits = self.metadataset.get_logits_from_probabilities(output)
        _, num_samples, num_classes, num_base_functions = X_val.shape

        val_loss = self.criterion(logits.reshape(-1, num_classes), y_val.reshape(-1)).item()

        return val_loss

    def fit_one_epoch(self, model, optimizer, batch_data):
        optimizer.zero_grad()
        X_batch, y_batch = batch_data[:2]
        num_samples, num_classes, num_base_functions = X_batch.shape
        output, w = model(X_batch)

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
        **kwargs,
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.dropout_rate = dropout_rate

        custom_weights_fc1 = torch.randn(
            output_dim
        )  # Custom weights for the first fully connected layer
        self.weight = nn.Parameter(custom_weights_fc1)

    def forward(
        self, x
    ):

        num_samples, num_classes, num_base_functions = x.shape
        base_functions = x
        mask = None
        if self.dropout_rate > 0 and self.training:
            mask= (torch.rand(size=(num_base_functions,)) > self.dropout_rate).float().to(x.device)
            for i, dim in enumerate([num_samples, num_classes]):
                mask = torch.repeat_interleave(
                    mask.unsqueeze(i), dim, dim=i
                )
            x = (x*mask)/(1-self.dropout_rate)
            base_functions = base_functions*mask

        w = torch.repeat_interleave(
            self.weight.reshape(1, -1), base_functions.shape[1], dim=0
        )

        w = w.reshape( num_samples, -1)
        
        if mask is not None and (not self.omit_output_mask):
            mask = mask.reshape(num_samples, -1)
            w = w.masked_fill(mask == 0, -1e9)
            
        w_norm = torch.nn.functional.softmax(w, dim=-1)
        w_norm = w_norm.reshape(
            num_samples, num_classes, num_base_functions
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
        inner_batch_size=10,
        dropout_dist=None,
        omit_output_mask=False,
        task_type="classification",
        mode="combined",
        num_classes=None,
        **kwargs
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.inner_batch_size = inner_batch_size
        self.dropout_dist = dropout_dist
        self.omit_output_mask = omit_output_mask
        self.mode = mode
        self.task_type = task_type
        self.num_classes = num_classes

        if self.mode == "model_averaging":
            num_layers=-1
        
        if self.mode == "stacking":
            output_dim = 1

        if self.add_y:
            input_dim += ( hidden_dim // 4 )
        
        self.class_embedding = nn.Embedding(num_classes, hidden_dim // 4)
        
        first_module = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers):
            first_module.append(nn.ReLU())
            first_module.append(nn.Linear(hidden_dim, hidden_dim))
        first_module.append(nn.ReLU())
        self.first_module = nn.Sequential(*first_module)
        self.second_module = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.ReLU())

        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate
        self.dropout_is_active = (dropout_rate > 0.) or (dropout_dist != None)


    def get_batched_weights(self, x):

        num_samples, num_classes, num_base_functions = x.shape
        x = x.transpose(0, 1).reshape(num_classes, num_samples, num_base_functions)

        # [NUMBER OF CLASSES X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS]
        x = self.first_module(x)
        x = x.transpose(0, 1) 

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
        self, x
    ):
        # X = [NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        num_samples, num_classes, num_base_functions = x.shape
        base_functions = copy.deepcopy(x)

        if self.dropout_is_active and self.training:
            mask, scaling_factor = self.get_mask_and_scaling_factor(num_base_functions, x.device)
            for i, dim in enumerate([num_samples, num_classes]):
                mask = torch.repeat_interleave(
                    mask.unsqueeze(i), dim, dim=i
                )
        else:
            mask = None

        w = []
        idx = np.arange(num_classes)
        for i in range(0, num_classes, self.inner_batch_size):
            range_idx = idx[range(i, min(i + self.inner_batch_size, num_classes))]
            if mask is not None:
                temp_x = (x[:,range_idx]*mask[:,range_idx])*scaling_factor
                base_functions[:,range_idx] = base_functions[:,range_idx]*mask[:,range_idx]
            else:
                temp_x = x[:,range_idx]
            
            if self.add_y:
                #class_indicator = torch.FloatTensor(range_idx).reshape(1,1,-1,1)*torch.ones(batch_size, num_samples, 1, 1)
                z = self.class_embedding(
                    torch.LongTensor(range_idx).to(temp_x.device)
                )
                z = torch.repeat_interleave(
                    z.unsqueeze(0), temp_x.shape[0], dim=0
                )
                temp_x = torch.cat([temp_x, z], axis=-1)
            temp_w = self.get_batched_weights(x = temp_x)
            w.append(temp_w)

        w = torch.cat(w, axis=1)

        if self.mode == "model_averaging":
            w = w.mean(axis=1) #same weight for all classes
            w = self.second_module(w)
            w = torch.repeat_interleave(
                w.unsqueeze(1), num_classes, dim=1
            )
            
        w = self.out_layer(w)

        if self.mode == "stacking":
            x = w.squeeze(-1)
            if self.task_type == "classification":
                x = torch.nn.functional.softmax(x, dim=-1)
            return x, None

        else:
            if (mask is not None) and (not self.omit_output_mask):
                w = w.masked_fill(mask == 0, -1e9)
            
            #batch_size, num_samples, num_classes, num_base_functions = w.shape
            if self.mode == "model_averaging":
                w_norm = torch.nn.functional.softmax(w, dim=-1)
                x = torch.multiply(base_functions, w_norm).sum(axis=-1)

            elif self.mode == "combined_conditional":
                w_norm = torch.nn.functional.softmax(w, dim=-1)
                x = torch.multiply(base_functions, w_norm).sum(axis=-1)

                if self.task_type == "classification":
                    norm_factor = x.sum(-1, keepdim=True) + 1e-10
                    x = torch.divide(x, norm_factor)               

            elif self.mode == "combined":
   
                w = w.reshape(num_samples, -1)
                w_norm = torch.nn.functional.softmax(w, dim=-1)
                w_norm = w_norm.reshape(
                    num_samples, num_classes, num_base_functions
                )
                x = torch.multiply(base_functions, w_norm).sum(axis=-1)

                if self.task_type == "classification":
                    norm_factor = x.sum(-1, keepdim=True) + 1e-10
                    x = torch.divide(x, norm_factor)  
            else:
                raise ValueError("Network mode is unknownod.")

            # x.shape: [BATCH_SIZE, NUM_SAMPLES, NUM_CLASSES]
            # w_norm.shape : [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
            return x, w_norm
