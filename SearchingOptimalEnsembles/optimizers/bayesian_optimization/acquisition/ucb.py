# Adapted from https://github.com/automl/neps/blob/master/src/neps/optimizers/bayesian_optimization/acquisition_functions/gp_ucb.py

import torch

from .base_acquisition import BaseAcquisition


class UpperConfidenceBound(BaseAcquisition):
    def __init__(
        self,
        device: torch.device,
        beta: float = 1.0,
        beta_decay: float = 0.95,
        in_fill: str = "best",
    ):
        """Calculates vanilla UCB over the candidate set.

        Args:
            beta: manual exploration-exploitation trade-off parameter.
            beta_decay: decay factor for beta
            in_fill: the criterion to be used for in-fill for the determination of incumbent value
                'best' means the empirical best observation so far (but could be
                susceptible to noise).
        """
        super().__init__(device=device)

        if in_fill not in ["best"]:
            raise ValueError(f"Invalid value for in_fill ({in_fill})")
        self.beta = beta
        self.beta_decay = beta_decay
        self.t = 0  # optimization trace size
        self.in_fill = in_fill
        self.incumbent = None

    def eval(self, x: torch.Tensor):
        """Evaluate the acquisition function at point x."""
        assert self.incumbent is not None, "UCB not fitted on model!!!"

        try:
            mean, stddev = self.surrogate_model.predict(x)
        except ValueError as e:
            raise e

        ucb = mean + torch.sqrt(self.beta) * stddev
        # with each round/batch of evaluation the exploration factor is reduced
        self.t += 1
        self.beta *= self.beta_decay**self.t

        return ucb

    def set_state(self, surrogate_model):
        super().set_state(surrogate_model)

        # TODO: verify min/max
        # Compute incumbent
        if self.in_fill == "best":
            _incumbent = [torch.min(y).item() for y in self.surrogate_model.y_obs]
            self.incumbent = min(_incumbent)
        else:
            raise NotImplementedError
