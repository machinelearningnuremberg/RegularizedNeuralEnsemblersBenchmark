from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution
from scipy.stats import norm

from ....utils.logger import get_logger


def input_to_torch(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        args = list(args)  # Convert args to a mutable list
        for i, arg in enumerate(args):
            if arg is not None:
                if isinstance(arg, np.ndarray):
                    args[i] = torch.tensor(
                        arg, dtype=torch.float32, device=self.device
                    )  # Convert numpy to torch.Tensor and move to device
                elif isinstance(arg, torch.Tensor):
                    args[i] = arg.to(
                        dtype=torch.float32, device=self.device
                    )  # Move torch.Tensor to device
        return method(self, *args, **kwargs)

    return wrapped


def input_to_numpy(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        args = [arg.cpu().numpy() if torch.is_tensor(arg) else arg for arg in args]
        return method(*args, **kwargs)

    return wrapped


def output_to_numpy(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        mu, stddev = method(self, *args, **kwargs)
        mu = mu.detach().to("cpu").numpy().reshape(-1)
        stddev = stddev.detach().to("cpu").numpy().reshape(-1)
        return mu, stddev

    return wrapped


@input_to_numpy
def EI(incumbent, mu, stddev):
    with np.errstate(divide="warn"):
        imp = mu - incumbent
        Z = imp / stddev
        score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)

    return score


def continuous_maximization(dim, bounds, acqf):
    result = differential_evolution(
        acqf,
        bounds=bounds,
        updating="immediate",
        workers=1,
        maxiter=20000,
        init="sobol",
    )
    return result.x.reshape(-1, dim)


class ConfigurableMeta(type):
    def __new__(mcs, name, bases, dct):
        if "default_config" not in dct:
            raise ValueError(f"{name} must have a default_config")
        return super().__new__(mcs, name, bases, dct)


class Checkpointer(ABC):
    def __init__(
        self,
        checkpoint_path: Path | None = None,
    ):
        self.logger = get_logger(name="SEO-CHECKPOINT", logging_level="debug")

        if checkpoint_path is None:
            self.logger.info("No checkpoint specified. Setting locally...")
            self.checkpoint_path = Path("checkpoints")  # TODO: fix it
        else:
            self.checkpoint_path = checkpoint_path

    def _checkpoint_exists(self, checkpoint_name: str = "checkpoint.pth") -> bool:
        """Checks if the checkpoint exists.

        Args:
            checkpoint_name (str, optional): Name of the checkpoint. Defaults to "checkpoint.pth".

        Returns:
            bool: True if the checkpoint exists.
        """
        if self.checkpoint_path is None:
            self.logger.info("No checkpoint specified. Skipping...")
            return False
        return (self.checkpoint_path / checkpoint_name).exists()

    def load_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """Loads the checkpoint from the checkpoint path."""
        if self._checkpoint_exists(checkpoint_name):
            checkpoint = self.checkpoint_path / checkpoint_name
            self.logger.info(f"Loading Checkpoint from {checkpoint_name}...")

            self._load_checkpoint(checkpoint)
        else:
            self.logger.info(f"Checkpoint {checkpoint_name} does not exist. Skipping...")
            return

    @abstractmethod
    def _load_checkpoint(self, checkpoint: Path = Path("surrogate.pth")):
        """Loads the checkpoint from the checkpoint path."""
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """Saves the checkpoint to the checkpoint path."""

        if self.checkpoint_path is None:
            self.logger.info("No checkpoint specified. Skipping...")
            return

        checkpoint = self.checkpoint_path / checkpoint_name
        self.logger.info(f"Saving Checkpoint to {checkpoint_name}...")
        self._save_checkpoint(checkpoint)

    @abstractmethod
    def _save_checkpoint(self, path: Path = Path("surrogate.pth")):
        """Saves the checkpoint to the checkpoint path."""
        raise NotImplementedError
