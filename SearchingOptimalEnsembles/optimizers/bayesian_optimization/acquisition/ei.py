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
        super().__init__(device=device)

        if in_fill not in ["best"]:
            raise ValueError(f"Invalid value for in_fill ({in_fill})")

        self.beta = beta
        self.in_fill = in_fill
        self.incumbent = None

    def eval(self, x: torch.Tensor):
        """Evaluate the acquisition function at point x.

        # TODO: add tensor dimensions


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

    def set_state(self, surrogate_model, **kwargs):
        super().set_state(surrogate_model, **kwargs)

        # TODO: verify min/max
        # Compute incumbent
        if self.in_fill == "best":
            _incumbent = [torch.min(y).item() for y in self.surrogate_model.y_obs]
            self.incumbent = min(_incumbent)
        else:
            raise NotImplementedError
