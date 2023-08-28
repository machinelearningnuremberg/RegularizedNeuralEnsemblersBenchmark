import gpytorch
import torch

# from typing import Callable

# from ....utils.common import instance_from_map

# KernelMapping: dict[str, Callable] = {
#     "rbf": gpytorch.kernels.RBFKernel,
#     "matern": gpytorch.kernels.MaternKernel,
# }


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        config: dict,
        dims: int,
    ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        kernel_function_mapping = {
            "rbf": self._create_scale_rbf_kernel,
            "matern": self._create_scale_matern_kernel,
        }

        kernel_name = config["kernel"].lower()

        if kernel_name not in kernel_function_mapping:
            raise ValueError(
                f"[ERROR] the kernel '{kernel_name}' is not supported for regression. "
                "Supported kernels are 'rbf' or 'matern'."
            )

        self.covar_module = kernel_function_mapping[kernel_name](config, dims)

    # def _create_kernel(self, kernel_name: str):

    #     kernel = instance_from_map(
    #         KernelMapping,
    #         kernel_name,
    #         name="kernel",
    #         kwargs=
    #     )

    #     return gpytorch.kernels.ScaleKernel(kernel)

    @staticmethod
    def _create_scale_rbf_kernel(config: dict, dims: int):
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dims if config["ard"] else None)
        )

    @staticmethod
    def _create_scale_matern_kernel(config: dict, dims: int):
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=config["nu"], ard_num_dims=dims if config["ard"] else None
            )
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
