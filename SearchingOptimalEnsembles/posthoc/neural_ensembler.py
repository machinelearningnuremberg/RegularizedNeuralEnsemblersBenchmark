from __future__ import annotations

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
    return (torch.log(w + 10e-8)).mean()


class NeuralEnsembler(BaseEnsembler):
    """Neural (End-to-End) Ensembler."""

    def __init__(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cuda"),
        prediction_device: torch.device = torch.device("cpu"),
        learning_rate=0.0001,
        epochs=1000,
        ne_hidden_dim: int = 512,
        ne_context_size: int = 32,
        ne_reg_term_div: float = 0.1,
        ne_add_y: bool = False,
        ne_eval_context_size: int = 50,
        ne_num_layers: int = 2,
        ne_use_context: bool = True,
        ne_reg_term_norm: float = 0.01,
        ne_net_type: str = "sas",
        ne_mode: Literal["inference", "pretraining"] = "inference",
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
        self.training = True
        self.predefined_pipeline_ids = None
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.epochs = epochs

    def set_state(
        self,
        metadataset: BaseMetaDataset,
        device: torch.device = torch.device("cuda"),
        mode: str = "inference",
        predefined_pipeline_ids: list[int] = None
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
        if self.use_context:

            if self.mode == "inference":
                actual_eval_context_size = min(
                    self.eval_context_size, self.metadataset.get_num_samples()
                )
            else:
                actual_eval_context_size = self.context_size
                               
                
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
            mask = mask.bool()
        #mask = None
        return mask

    def get_context(self, X_obs, metadataset):
        # X_context are the base functions for val dataset for a subset fo samples
        # y_context are the
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
        if self.net_type == "bas":
            idx = np.random.randint(0, num_base_functions, self.context_size)
            return (X_train[..., idx], base_functions_train[..., idx], y_train)

        elif self.net_type == "sas":
            idx = np.random.randint(0, num_samples, self.context_size)
            return (X_train[:, idx], base_functions_train[:, idx], y_train[:, idx])

        else:
            return (X_train, base_functions_train, y_train, X_context, y_context, mask_context)
        
    def get_batch_for_pretraining(self):
        dataset_name = np.random.choice(self.metadataset.dataset_names, 1).item()
        self.metadataset.set_state(dataset_name=dataset_name)
        num_samples = self.metadataset.get_num_samples()
        samples_idx = np.random.randint(0, num_samples, self.context_size)

        #X and base functions are the same
        X = base_functions = (
            self.metadataset
            .get_predictions([self.predefined_pipeline_ids])[0]
            .transpose(0, 1)
            .transpose(2, 1)
            .unsqueeze(0)
        )[:,samples_idx]
        y = self.metadataset.get_targets()[samples_idx].unsqueeze(0).to(self.device)

        X_context, y_context = self.get_context(self.predefined_pipeline_ids, self.metadataset)
        mask_context = self.get_mask_context()

        return self.send_to_device(X, base_functions, y, X_context, y_context, mask_context)
    
    def pretrain_net(self, predefined_pipeline_ids: list[int],
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

    def fit_net(
        self,
        X_train,
        y_train,
        base_functions_train
    ):
        self.training = True
        if self.net_type == "bas":
            NetClass = ENetBAS
            input_dim = base_functions_train.shape[-2]  # NUMBER OF CLASSES
            output_dim = 1
        elif self.net_type == "sas":
            NetClass = ENetSAS
            input_dim = X_train.shape[-1]
            output_dim = base_functions_train.shape[-1]  # [NUMBER OF BASE FUNCTIONS]
        elif self.net_type == "cas":
            NetClass = ENetCAS
            input_dim = base_functions_train.shape[-1]   
            output_dim =  base_functions_train.shape[-1]  
        elif self.net_type == "ps":
            NetClass = ENetPS
            input_dim = base_functions_train.shape[-1]
            output_dim = base_functions_train.shape[-1]  
        elif self.net_type == "simple":
            NetClass = ENetSimple
            input_dim = 1
            output_dim = base_functions_train.shape[-1]*base_functions_train.shape[-2]      
        else:
            raise NotImplementedError()

        net = NetClass(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            add_y=self.add_y,
            num_layers=self.num_layers,
        )
        y_train = torch.tensor(y_train, dtype=torch.long)
        optimizer = Adam(net.parameters(), lr=self.learning_rate)
        net.train()

        X_train, y_train, base_functions_train, net = self.send_to_device(
            X_train, y_train, base_functions_train, net
        )

        for epoch in range(self.epochs):
            optimizer.zero_grad()
    
            if self.mode == "inference":
                batch_data = self.get_batch(X_train, base_functions_train, y_train)
            elif self.mode == "pretraining":
                batch_data = self.get_batch_for_pretraining()
                batch_data = self.send_to_device(*batch_data)
            else:
                raise NotImplementedError
            
            loss, div, l1 = self.fit_one_epoch(net, optimizer, batch_data)
            print("Epoch", epoch, "Loss", loss.item(), "Div Loss", div.item())

        net.eval()
        self.training = False
        return net

    def fit_one_epoch(self, net, optimizer, batch_data):
        X_batch, base_functions_batch, y_batch = batch_data[:3]
        _, num_samples, num_classes, num_base_functions = X_batch.shape
        output, w = net(*batch_data)
        logits = self.metadataset.get_logits_from_probabilities(output)
        loss = self.criterion(logits.reshape(-1, num_classes), y_batch.reshape(-1))
        div = div_loss(w, base_functions_batch)
        l1 = l1_norm(w)
        loss -= self.reg_term_div * div
        loss += self.reg_term_norm * l1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        return loss, l1, div

    def load_model_from_checkpoint(self, checkpoint: str = "neural_ensembler.pt"):
        raise NotImplementedError()
        

class ENetPS(nn.Module): # Parallel Samples
    def __init__(
        self,
        input_dim=1, #number of pipelines
        hidden_dim=128,
        output_dim=1, #number of pipelines
        num_layers=2,
        num_heads=1,
        add_y=True,
        mask_prob=0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.mask_prob = mask_prob 
        
        self.ff = FeedforwardNetwork(input_dim=input_dim, 
                                     output_dim=output_dim,
                                     num_hidden_layers=num_layers,
                                     hidden_dim=hidden_dim)
        
    def forward(
        self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None
    ):
        # x = [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        batch_size, num_samples, num_classes, num_base_functions = x.shape

        x = self.ff(x)
 
        w = x.reshape(batch_size, num_samples, -1)
        w_norm = torch.nn.functional.softmax(w, dim=-1)
        w_norm = w_norm.reshape(batch_size, num_samples, num_classes, num_base_functions)

        x = torch.multiply(base_functions, w_norm).sum(axis=-1)
        # x.shape: [BATCH_SIZE X NUM_SAMPLES, NUM_CLASSES]
        # w_norm.shape : [BATCH_SIZE X NUM_SAMPLES  X NUM_CLASSES X NUMBER OF BASE FUNCTIONS]
        return x, w_norm       


class ENetCAS(nn.Module): # Class as Sequence
    def __init__(
        self,
        input_dim=1, #number of pipelines
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        num_heads=1,
        add_y=True,
        mask_prob=0.5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.mask_prob = mask_prob

        self.embedding = nn.Linear(input_dim, hidden_dim)
        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None
    ):
        # x = [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        batch_size, num_samples, num_classes, num_base_functions = x.shape

        x = x.reshape(-1, num_classes, num_base_functions)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.output_layer(x)  # ((BATCH_SIZE X NUM_SAMPLES) X NUM_BASE_FUNCTIONS)
        x = x.unsqueeze(0) #only valid for batch size =1 
        x = x.reshape((batch_size, num_samples, num_classes, num_base_functions))
        #x = torch.repeat_interleave(x.unsqueeze(2), num_classes, dim=2)

        w = x.reshape(batch_size, num_samples, -1)
        w_norm = torch.nn.functional.softmax(w, dim=-1)
        w_norm = w_norm.reshape(batch_size, num_samples, num_classes, num_base_functions)

        x = torch.multiply(base_functions, w_norm).sum(axis=-1)
        # x.shape: [BATCH_SIZE X NUM_SAMPLES, NUM_CLASSES]
        # w_norm.shape : [BATCH_SIZE X NUM_SAMPLES  X NUM_CLASSES X NUMBER OF BASE FUNCTIONS]
        return x, w_norm



class ENetBAS(nn.Module):  # Base model as sequence
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        num_heads=1,
        add_y=True,
        mask_prob=0.5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.mask_prob = mask_prob

        self.embedding = nn.Linear(input_dim, hidden_dim)
        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None
    ):
        # x = [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        batch_size, num_samples, num_classes, num_base_functions = x.shape

        x = x.transpose(2,3).reshape(-1, num_base_functions, num_classes)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.output_layer(x)  # ((BATCH_SIZE X NUM_SAMPLES) X NUM_BASE_FUNCTIONS)
        x = x.reshape((batch_size, num_samples, num_base_functions))
        x = torch.repeat_interleave(x.unsqueeze(2), num_classes, dim=2)

        w = x.reshape(batch_size, num_samples, -1)
        w_norm = torch.nn.functional.softmax(w, dim=-1)
        w_norm = w_norm.reshape(batch_size, num_samples, num_classes, num_base_functions)

        x = torch.multiply(base_functions, w_norm).sum(axis=-1)
        # x.shape: [BATCH_SIZE X NUM_SAMPLES, NUM_CLASSES]
        # w_norm.shape : [BATCH_SIZE X NUM_SAMPLES  X NUM_CLASSES X NUMBER OF BASE FUNCTIONS]
        return x, w_norm



class ENetSimple(nn.Module): 
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        output_dim=1,
        num_layers=2,
        simple_coefficients=False,
        dropout_rate=0,
        num_heads=1,
        add_y=False,
        mask_prob=0.5,
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.mask_prob = mask_prob

        custom_weights_fc1 = torch.randn(
            output_dim
        )  # Custom weights for the first fully connected layer
        self.weight = nn.Parameter(custom_weights_fc1)

    def forward(
        self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None
    ):
        
        batch_size, num_samples, num_classes, num_base_functions = x.shape
        x = torch.repeat_interleave(
            self.weight.reshape(1, -1), base_functions.shape[1], dim=0
        )

        w = x
        w = w.reshape(batch_size, num_samples, -1)
        w_norm = torch.nn.functional.softmax(w, dim=-1)
        w_norm = w_norm.reshape(
            batch_size, num_samples, num_classes, num_base_functions
        )
        x = torch.multiply(base_functions, w_norm).sum(axis=-1)
              
        return x, w_norm


class ENetSAS(nn.Module):  # Sample as Sequence
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
        inner_batch_size=50
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_y = add_y
        self.mask_prob = mask_prob
        self.inner_batch_size = inner_batch_size

        if self.add_y:
            input_dim += 1

        self.embedding = nn.Linear(input_dim, hidden_dim)

        first_encoder_modules = [
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
            )
            for _ in range(num_layers)
        ]
        self.first_encoder = nn.ModuleList(first_encoder_modules)

        self.second_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )

        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)


    def get_batched_weights(self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None):
        
        batch_size, num_samples, num_classes, num_base_functions = x.shape

        if X_context is not None:
            x = torch.cat([X_context, x], dim=1)

        if self.add_y:
            if y is None or y_context is not None:
                y = -1*torch.ones(batch_size, num_samples, num_classes, 1).to(x.device)
            else:
                y = nn.functional.one_hot(y, num_classes=num_classes).unsqueeze(3)

            if y_context is not None:
                y_context = nn.functional.one_hot(y_context, num_classes=num_classes).unsqueeze(3)
                y = torch.cat([y_context, y], dim=1)

            x = torch.cat([x,y], dim=-1)
        
        batch_size, num_samples, num_classes, num_base_functions = x.shape
        x = x.transpose(1, 2).reshape(-1, num_samples, num_base_functions)
        
        # [(BATCH SIZE X NUMBER OF CLASSES) X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS]
        x = self.embedding(x)
        for encoder in self.first_encoder:
            x = encoder(x, src_mask=mask_context)
        
        x = self.second_encoder(x, src_mask=mask_context)
        x = self.out_layer(x)
        x = x.transpose(0, 1).unsqueeze(0)  # use batch size here

        return x

    def forward(
        self, x, base_functions, y=None, X_context=None, y_context=None, mask_context=None
    ):
        # X = [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        batch_size, num_samples, num_classes, num_base_functions = x.shape
        num_query_samples = x.shape[1]
        
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

        #num_classes changed
        batch_size, num_samples, num_classes, num_base_functions = w.shape
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
