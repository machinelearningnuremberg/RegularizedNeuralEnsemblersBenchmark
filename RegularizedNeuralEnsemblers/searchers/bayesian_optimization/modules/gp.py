from typing import Callable

import gpytorch
import torch

from ....utils.common import instance_from_map

KernelMapping: dict[str, Callable] = {
    "rbf": gpytorch.kernels.RBFKernel,
    "matern": gpytorch.kernels.MaternKernel,
}


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        dims: int,
        kernel_name: str = "matern",
        ard: bool = False,
        nu: float = 2.5,
    ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_name not in KernelMapping:
            raise ValueError(
                f"[ERROR] the kernel '{kernel_name}' is not supported for regression. "
                "Supported kernels are 'rbf' or 'matern'."
            )

        self.covar_module = self._create_kernel(
            kernel_name=kernel_name, ard=ard, nu=nu, dims=dims
        )

    @staticmethod
    def _create_kernel(kernel_name: str, ard, nu, dims: int):
        kernel_kwargs = {"ard_num_dims": dims if ard else None}
        if kernel_name == "matern":
            kernel_kwargs["nu"] = nu

        kernel = instance_from_map(
            KernelMapping,
            kernel_name,
            name="kernel",
            kwargs=kernel_kwargs,
        )

        return gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
