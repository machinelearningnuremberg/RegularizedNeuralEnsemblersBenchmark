from abc import ABC, abstractmethod

import torch


class BaseAcquisition(ABC):
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.surrogate_model = None
        self.device = device

    @abstractmethod
    def eval(self, x: torch.Tensor):
        """Evaluate the acquisition function at point x."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def set_state(self, surrogate_model, **kwargs):  # pylint: disable=unused-argument
        self.surrogate_model = surrogate_model
