from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from typing_extensions import Literal

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..samplers.random_sampler import RandomSampler
from .base_ensembler import BaseEnsembler

# command to test:
# python main.py --apply_posthoc_ensemble_at_end --ensembler_name neural --data_version micro --ne_use_context --project_name SOE --num_iterations 100 --metric_name error --dataset_id 2 --run_name neural45_2_4 --meta_split_id 4 --searcher_name None --ne_hidden_dim 128 --ne_context_size 8 --ne_reg_term_div 0.1 --ne_eval_context_size 32 --experiment_group neural45_2 --no_wandb

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


def l1_norm(w):
    # return torch.norm(w, p=1, dim=-1).mean()
    return (w*torch.log(w + 10e-8)).mean()


class MTNeuralEnsembler(BaseEnsembler):
    """Neural (End-to-End) Ensembler."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cuda"),
        prediction_device: torch.device = torch.device("cpu"),
        learning_rate: float =0.0001,
        epochs: int = 1000,
        use_wandb: bool = False,
        checkpoint_freq: int = 1000,
        run_name: str = None,
        num_pipelines: int = 64,
        project_name: str = "Training_NE",
        ne_hidden_dim: int = 512,
        ne_context_size: int = 128,
        ne_reg_term_div: float = 0.1,
        ne_add_y: bool = True,
        ne_eval_context_size: int = 128,
        ne_num_layers: int = 2,
        ne_use_context: bool = True,
        ne_reg_term_norm: float = 0.0,
        ne_net_type: str = "sas",
        ne_num_heads: int = 4,
        ne_mode: Literal["inference", "pretraining"] = "inference",
        ne_checkpoint_name: str = "auto",
        ne_resume_from_checkpoint: bool = False,
        ne_use_mask: bool = True,
        ne_unique_weights_per_function: bool = False,
        ne_dropout_rate: float = 0.,
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
        self.reg_term_norm = ne_reg_term_norm
        self.net_type = ne_net_type
        self.prediction_device = prediction_device
        self.mode = ne_mode

        self.learning_rate = learning_rate
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

        self.project_name = project_name # for wandb
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

    def update_context_size(
        self, context_size: int = None
    ):
        self.context_size = context_size

    def batched_prediction(
        self, X, base_functions, y=None, X_context=None, y_context=None, mask_context=None
    ):

        _, num_samples, num_classes, num_pipelines = X.shape
        idx = np.arange(num_samples)
        # np.random.shuffle(idx)
        outputs = []
        weights = []

        if self.task_type == "regression":
            #base_functions is not necessary to transform because they permit to mantain the original scale
            X /= self.y_max

            if y is not None:
                y /= self.y_max
            
            if y_context is not None:
                y_context /= self.y_max

            if  X_context is not None:
                X_context /= self.y_max

        for i in range(0, num_samples, self.eval_context_size):
            range_idx = idx[range(i, min(i + self.eval_context_size, num_samples))]

            if mask_context is not None: #assuming mask is only valid in eval
                temp_mask = mask_context[
                    : (self.context_size + len(range_idx)),
                    : (self.context_size + len(range_idx)),
                ]
            else:
                temp_mask = None
            temp_y = y[:, range_idx] if y is not None else None

            with torch.no_grad():
                output, w = self.net(
                    x=X[:, range_idx],
                    base_functions=base_functions[:, range_idx],
                    y=temp_y,
                    X_context=X_context,
                    y_context=y_context,
                    mask_context=temp_mask,
                )
            w = w.transpose(2, 3).transpose(
                1, 2
            )  # Expected shape [BATCH SIZE X NUM_PIPELINES X NUM_SAMPLES X NUM_CLASSES]
            outputs.append(output.to(self.prediction_device))
            weights.append(w.to(self.prediction_device))

        return torch.cat(outputs, axis=-2).to(self.prediction_device), torch.cat(
            weights, axis=-2
        ).to(self.prediction_device)

    def get_weights(self, X_obs, X_context=None, y_context=None):
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
            X=base_functions,
            base_functions=base_functions,
            X_context=X_context,
            y_context=y_context,
            mask_context=mask_context,
        )
        return weights

    def get_mask_context(self):
        mask = None
        if self.use_context and self.use_mask:
            if not self.training:
                actual_eval_context_size = min(
                    self.eval_context_size, self.metadataset.get_num_samples()
                )

            else:
                actual_eval_context_size = self.eval_context_size


            mask_upper_left_tile = torch.ones(self.context_size, self.context_size)
            mask_upper_right_tile = torch.zeros(
                self.context_size, actual_eval_context_size
            )
            mask_lower_left_tile = torch.ones(actual_eval_context_size, self.context_size)
            mask_lower_right_tile = torch.eye(actual_eval_context_size)
            # mask_lower_right_tile = torch.ones(actual_eval_context_size, actual_eval_context_size)
            mask_upper = torch.cat([mask_upper_left_tile, mask_upper_right_tile], dim=1)
            mask_lower = torch.cat([mask_lower_left_tile, mask_lower_right_tile], dim=1)
            mask = torch.cat([mask_upper, mask_lower], dim=0).to(self.device)
            mask = 1-mask
            mask = mask.bool()
        
        return mask

    def get_context(self, X_obs, metadataset = None):
        # X_context are the base functions for val dataset for a subset fo samples
        # y_context are the

        if metadataset is None:
            metadataset = self.metadataset
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
            X_context = base_functions[:, samples_idx_for_context].to(self.device)
            y_context = (
                metadataset.get_targets()[samples_idx_for_context]
                .unsqueeze(0)
                .to(self.device)
            )

        return X_context, y_context

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
        self.net = self.fit_net(
            X_train=base_functions, y_train=y, base_functions_train=base_functions
        )
        _, weights = self.batched_prediction(
            X=base_functions, base_functions=base_functions, y=y
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
        X_context = y_context = mask_context = None
    
        if self.net_type == "sas":
            #if self.use_context:
            #    X_context, y_context = self.get_context(self.X_obs, self.metadataset)
            #    mask_context = self.get_mask_context()
                
            idx = np.random.randint(0, num_samples,self.eval_context_size)
            return (X_train[:, idx], base_functions_train[:, idx], y_train[:, idx],  X_context, y_context, mask_context)

        else:
            return (X_train, base_functions_train, y_train, X_context, y_context, mask_context)

    def get_batch_for_pretraining(self, meta_split="meta-train"):
        dataset_name = np.random.choice(self.metadataset.meta_splits[meta_split], 1).item()
        self.metadataset.set_state(dataset_name=dataset_name)
        num_samples = self.metadataset.get_num_samples()
        samples_idx = np.random.randint(0, num_samples, self.context_size)

        if self.predefined_pipeline_ids is not None:
            predefined_pipeline_ids = self.predefined_pipeline_ids
        else:
            predefined_pipeline_ids = np.random.randint(0, self.num_pipelines, self.metadataset.get_num_pipelines())

        #X and base functions are the same
        X = base_functions = (
            self.metadataset
            .get_predictions([predefined_pipeline_ids])[0]
            .transpose(0, 1)
            .transpose(2, 1)
            .unsqueeze(0)
        )[:,samples_idx]
        y = self.metadataset.get_targets()[samples_idx].unsqueeze(0).to(self.device)

        X_context, y_context = self.get_context(predefined_pipeline_ids, self.metadataset)
        mask_context = self.get_mask_context()

        return self.send_to_device(X, base_functions, y, X_context, y_context, mask_context)

    def pretrain_net(self, predefined_pipeline_ids: list[int] = None,
                      pretrain_learning_rate: float = 0.0001,
                      pretrain_epochs: int = 1000):
        
        self.predefined_pipeline_ids = predefined_pipeline_ids
        X, base_functions, y = self.get_batch_for_pretraining()[:3]
        self.mode = "pretraining"
        finetuning_learning_rate = self.learning_rate
        finetuning_epochs = self.epochs
        self.learning_rate = pretrain_learning_rate
        self.epochs = pretrain_epochs
        self.fit_net(X, y, base_functions)
        self.learning_rate = finetuning_learning_rate
        self.epochs = finetuning_epochs
        self.mode = "inference"

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
            dropout_rate=dropout_rate
        )
        best_val_loss = np.inf

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
            optimizer.zero_grad()

            if self.mode == "inference":
                batch_data = self.get_batch(X_train, base_functions_train, y_train)
            elif self.mode == "pretraining":
                batch_data = self.get_batch_for_pretraining()
                batch_data = self.send_to_device(*batch_data)
            else:
                raise NotImplementedError

            loss, div, l1, w = self.fit_one_epoch(model, optimizer, batch_data)
            print("Epoch", epoch, "Loss", loss.item(), "Div Loss", div.item(), "W:", w.median().item())

            if self.mode == "pretraining":
                if self.use_wandb:
                    wandb.log({"epoch": epoch,
                            "loss": loss.item(),
                            "div_loss": div.item(),
                            "l1_loss": l1.item(),
                            "w_mean": w.mean(),
                            "w_max": w.max(),
                            "w_min": w.min(),
                            "w_std": w.std(),
                            "w_mean_max": w.max(-2)[0].mean()
                            }
                            )

                if (epoch % self.checkpoint_freq) == 0:                    
                    val_loss = self.meta_validate(model)
                    if val_loss < best_val_loss:
                        self.save_checkpoint(model, optimizer, epoch, loss.item())
                    best_val_loss = val_loss
                    print("Epoch", epoch, "Val Loss", val_loss)
        model.eval()

        if self.mode == "pretraining":
            self.save_checkpoint(model, optimizer, epoch, loss.item())

        self.training = False
        return model

    def validate(self, model, X_val, y_val, base_functions_val):

        X_val, y_val, base_functions_val = self.send_to_device(X_val, y_val, base_functions_val)
        output, w = model(X_val, base_functions_val)
        logits = self.metadataset.get_logits_from_probabilities(output)
        _, num_samples, num_classes, num_base_functions = X_val.shape

        val_loss = self.criterion(logits.reshape(-1, num_classes), y_val.reshape(-1)).item()

        return val_loss

    def meta_validate(self, model, validation_iterations=100):
        val_loss = 0
        for i in range(validation_iterations):
            batch_data = self.get_batch_for_pretraining(meta_split="meta-valid")
            X_batch, base_functions_batch, y_batch = batch_data[:3]
            _, num_samples, num_classes, num_base_functions = X_batch.shape

            with torch.no_grad():
                output, w = model(*batch_data)
                logits = self.metadataset.get_logits_from_probabilities(output)
                loss = self.criterion(logits.reshape(-1, num_classes), y_batch.reshape(-1)).item()
                val_loss += loss
        val_loss /= validation_iterations

        if self.use_wandb and self.mode=="pretraining":
            wandb.log({"meta_val_loss": val_loss
                    }
                )
        return val_loss

    def fit_one_epoch(self, model, optimizer, batch_data):
        X_batch, base_functions_batch, y_batch = batch_data[:3]
        _, num_samples, num_classes, num_base_functions = X_batch.shape
        output, w = model(*batch_data)

        if self.task_type == "classification":
            logits = self.metadataset.get_logits_from_probabilities(output)
            loss = self.criterion(logits.reshape(-1, num_classes), y_batch.reshape(-1))
            div = div_loss(w, base_functions_batch)
            l1 = l1_norm(w)
            loss -= self.reg_term_div * div
            loss += self.reg_term_norm * l1
        else: #assuming task is regression
            loss = self.criterion(output.reshape(-1), y_batch.reshape(-1))
            l1 = torch.tensor([0])
            div = torch.tensor([0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        return loss, l1, div, w

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
        self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None
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

        # w = w.reshape(
        #     batch_size, num_samples, num_classes, num_base_functions
        # )
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
        unique_weight_per_base_function=True
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


    def get_batched_weights(self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None):

        batch_size, num_samples, num_classes, num_base_functions = x.shape
        x = x.transpose(1, 2).reshape(-1, num_samples, num_base_functions)

        # [(BATCH SIZE X NUMBER OF CLASSES) X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS]
        x = self.embedding(x)
        for encoder in self.first_encoder:
            x = encoder(x)

        #x = self.second_encoder(x, src_mask=mask_context)
        x = x.transpose(0, 1).unsqueeze(0)  # use batch size here

        return x

    def forward(
        self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None
    ):
        # X = [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        batch_size, num_samples, num_classes, num_base_functions = x.shape
        num_query_samples = x.shape[1]

        if self.dropout_rate > 0 and self.training:
            mask= (torch.rand(size=(num_base_functions,)) > self.dropout_rate).float().to(x.device)
            for i, dim in enumerate([batch_size, num_samples, num_classes]):
                mask = torch.repeat_interleave(
                    mask.unsqueeze(i), dim, dim=i
                )
            x = (x*mask)/(1-self.dropout_rate)
            base_functions = base_functions*mask
        else:
            mask = None

        w = []
        idx = np.arange(num_classes)
        for i in range(0, num_classes, self.inner_batch_size):
            range_idx = idx[range(i, min(i + self.inner_batch_size, num_classes))]
            temp_w = self.get_batched_weights(x = x[:,:,range_idx],
                                              base_functions = base_functions[:,:,range_idx],
                                               y= y,
                                               X_context=X_context[:,:,range_idx] if X_context is not None else None,
                                               y_context=y_context,
                                               mask_context=mask_context)
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
        batch_size, num_samples, num_classes, num_base_functions = w.shape
        
        if mask is not None:
            w = w.masked_fill(mask == 0, -1e9)
        w = w.reshape(batch_size, num_samples, -1)
        w_norm = torch.nn.functional.softmax(w, dim=-1)
        w_norm = w_norm.reshape(
            batch_size, num_samples, num_classes, num_base_functions
        )
        w_norm = w_norm[:, -num_query_samples:]

        x = torch.multiply(base_functions, w_norm).sum(axis=-1)
        # x.shape: [BATCH_SIZE, NUM_SAMPLES, NUM_CLASSES]
        # w_norm.shape : [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        return x, w_norm
