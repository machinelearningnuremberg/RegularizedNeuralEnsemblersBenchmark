# Adapted from https://github.com/automl/neps/blob/master/src/neps/optimizers/bayesian_optimization/acquisition_functions/ei.py


import torch
from torch.distributions import Normal

from .base_acquisition import BaseAcquisition


class ExpectedImprovement(BaseAcquisition):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        beta: float = 0.0,
    ):
        """
        Expected Improvement (EI) acquisition function.

        Args:
            beta: manual exploration-exploitation trade-off parameter.
        """
        super().__init__(device=device)

        self.beta = beta
        self.incumbent = None

    def eval(
        self,
        x: torch.Tensor,
        metric_per_pipeline: torch.Tensor = None,
        max_num_pipelines: int = 10,
    ) -> torch.Tensor:
        """
        Evaluate the acquisition function at a given point x.

        Args:
            x : torch.Tensor
                The input tensor to evaluate the acquisition function.
                Shape should be (B, N, F) where:
                - B is the batch size
                - N is the number of pipelines
                - F is the number of features
            metric_per_pipeline : torch.Tensor
                Metric of the pipelines, shape (B, N) where B is the batch size and N is
                the number of pipelines.
            max_num_pipelines : int
                Maximum number of pipelines to use for the acquisition function.

        Returns:
            torch.Tensor
                Expected Improvement (EI) at point x (actually negative EI as we always minimize!!).

        Raises:
            AssertionError
                If self.incumbent is not set or None.

            ValueError
                If there's an error when predicting using the surrogate model.

        """

        assert self.incumbent is not None, "EI not fitted on model!!!"

        try:
            mean, stddev = self.surrogate_model.predict(
                x=x,
                metric_per_pipeline=metric_per_pipeline,
                max_num_pipelines=max_num_pipelines,
            )
        except ValueError as e:
            raise e

        gauss = Normal(
            torch.zeros(1, device=self.device), torch.ones(1, device=self.device)
        )

        imp = self.incumbent - mean
        u = (imp - self.beta) / stddev
        ucdf = gauss.cdf(u)
        updf = torch.exp(gauss.log_prob(u))
        ei = stddev * updf + (imp - self.beta) * ucdf

        return -ei

    def set_state(self, surrogate_model, incumbent, **kwargs):
        """
        Set the state of the acquisition function.

        Args:
            surrogate_model: the surrogate model to be used for the acquisition function.

        """
        super().set_state(surrogate_model)
        self.incumbent = incumbent
