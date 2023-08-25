# Adapted from https://github.com/automl/neps/blob/master/src/neps/optimizers/bayesian_optimization/acquisition_functions/ei.py


import torch
from torch.distributions import Normal

from .base_acquisition import BaseAcquisition


class ExpectedImprovement(BaseAcquisition):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        beta: float = 0.0,
        in_fill: str = "best",
    ):
        """
        Expected Improvement (EI) acquisition function.

        Args:
            beta: manual exploration-exploitation trade-off parameter.
            in_fill: the criterion to be used for in-fill for the determination of incumbent value
                'best' means the empirical best observation so far (but could be
                susceptible to noise).
        """
        super().__init__(device=device)

        if in_fill not in ["best"]:
            raise ValueError(f"Invalid value for in_fill ({in_fill})")

        self.beta = beta
        self.in_fill = in_fill
        self.incumbent = None

    def eval(self, x: torch.Tensor):
        """
        Evaluate the acquisition function at a given point x.

        Args:
            x : torch.Tensor
                The input tensor to evaluate the acquisition function.
                Shape should be (B, N, F) where:
                - B is the batch size
                - N is the number of pipelines
                - F is the number of features

        Returns:
            torch.Tensor
                Expected Improvement (EI) at point x.

        Raises:
            AssertionError
                If self.incumbent is not set or None.

            ValueError
                If there's an error when predicting using the surrogate model.

        """

        assert self.incumbent is not None, "EI not fitted on model!!!"

        try:
            mean, stddev = self.surrogate_model.predict(x)
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

        return ei

    def set_state(self, surrogate_model):
        """
        Set the state of the acquisition function.

        Args:
            surrogate_model: the surrogate model to be used for the acquisition function.

        """
        super().set_state(surrogate_model)

        # TODO: verify min/max
        # Compute incumbent
        if self.in_fill == "best":
            _incumbent = [torch.min(y).item() for y in self.surrogate_model.y_obs]
            self.incumbent = min(_incumbent)
        else:
            raise NotImplementedError
