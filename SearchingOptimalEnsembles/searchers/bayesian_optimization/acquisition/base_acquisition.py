from abc import ABC, abstractmethod

import torch

from ....utils.logger import get_logger


class BaseAcquisition(ABC):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Base class for acquisition functions.

        Args:
            device: torch.device
                The device to use for the acquisition function.
        """
        self.surrogate_model = None
        self.device = device
        self.logger = get_logger(name="SEO-ACQUISITION", logging_level="debug")

    @abstractmethod
    def eval(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Evaluate the acquisition function at a given point x.

        Args:
            x: torch.Tensor
                The input tensor to evaluate the acquisition function.
                Shape should be (B, N, F) where:
                - B is the batch size
                - N is the number of pipelines
                - F is the number of features

        Returns:
            torch.Tensor
                acquisition function value at point x.

        """

        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def set_state(self, surrogate_model, **kwargs):  # pylint: disable=unused-argument
        """
        Set the state of the acquisition function.

        Args:
            surrogate_model: BaseSurrogateModel
                The surrogate model to use for the acquisition function.

        """
        self.surrogate_model = surrogate_model
