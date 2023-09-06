import os
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

get_arch_str = lambda op_list: "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*op_list)
get_arch_filename = lambda x: "0" * (6 - len(x)) + x


class Baselearner:
    """
    A container class for baselearner networks which can hold the nn.Module,
    predictions (as tensors) and evaluations. It has methods for computing the predictions
    and evaluations on validation and test sets.
    """

    _cpu_device = torch.device("cpu")

    def __init__(
        self,
        model_id,
        net_config=None,
        device=None,
        nn_module=None,
        preds=None,
        evals=None,
    ):
        self.model_id = model_id
        self.net_config = net_config
        self.device = device
        self.nn_module = nn_module
        self.preds = preds
        self.evals = evals

    def to_device(self, device=None):
        if device is None:
            device = self._cpu_device

        if self.nn_module is not None:
            self.nn_module.to(device)

        if self.preds is not None:
            for dct in self.preds.values():
                for k, tsr_dst in dct.items():
                    dct[k] = TensorDataset(
                        tsr_dst.tensors[0].to(device), tsr_dst.tensors[1].to(device)
                    )

        self.device = device

    def partially_to_device(self, data_type, device=None):
        assert data_type in ["val", "test"], "Invalid data_type arg."

        if device is None:
            device = self._cpu_device

        if self.nn_module is not None:
            self.nn_module.to(device)

        if self.preds is not None:
            dct = self.preds[data_type]  # only move the requested data_type to gpu
            for k, tsr_dst in dct.items():
                dct[k] = TensorDataset(
                    tsr_dst.tensors[0].to(device), tsr_dst.tensors[1].to(device)
                )

        self.device = device

    def save(self, directory, filename, force_overwrite=False):
        self.to_device(self._cpu_device)

        if force_overwrite:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Forcefully overwriting {directory}")
        else:
            Path(directory).mkdir(parents=True, exist_ok=False)

        torch.save(
            {"config": self.net_config, "preds": self.preds, "evals": self.evals},
            os.path.join(directory, f"{filename}.pt"),
        )

    @classmethod
    def load(cls, data_dir: str, filename: str, load_nn_module: bool = False):
        config, preds, evals = torch.load(
            os.path.join(data_dir, f"{filename}.pt")
        ).values()

        if load_nn_module:
            nn_module = torch.load(os.path.join(data_dir, "nn_module.pt"))
        else:
            nn_module = None

        device = cls._cpu_device
        obj = cls(
            model_id=filename,
            net_config=config,
            device=device,
            nn_module=nn_module,
            preds=preds,
            evals=evals,
        )
        return obj
