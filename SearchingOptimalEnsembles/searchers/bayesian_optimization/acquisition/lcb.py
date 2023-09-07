# Adapted from https://github.com/automl/neps/blob/master/src/neps/optimizers/bayesian_optimization/acquisition_functions/gp_ucb.py


import torch

from .base_acquisition import BaseAcquisition


class LowerConfidenceBound(BaseAcquisition):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        beta: float = 1.0,
        beta_decay: float = 1.0,
    ):
        """Calculates vanilla LCB over the candidate set.

        Args:
            beta: manual exploration-exploitation trade-off parameter.
            beta_decay: decay factor for beta
        """
        super().__init__(device=device)

        self.beta = torch.FloatTensor([beta]).to(device)
        self.beta_decay = beta_decay
        self.t = 0  # optimization trace size
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
            x: torch.Tensor
                The input tensor to evaluate the acquisition function.
                Shape should be (B, N, F) where:
                - B is the batch size
                - N is the number of pipelines
                - F is the number of features
            metric_per_pipeline: torch.Tensor
                Metric of the pipelines, shape (B, N) where B is the batch size and N is
                the number of pipelines.
            max_num_pipelines: int
                Maximum number of pipelines to use for the acquisition function.

        Returns:
            torch.Tensor
                Lower Confidence Bounds (LCB) at point x.

        Raises:
            AssertionError
                If self.incumbent is not set or None.

            ValueError
                If there's an error when predicting using the surrogate model.

        """
        assert self.incumbent is not None, "LCB not fitted on model!!!"

        try:
            mean, stddev = self.surrogate_model.predict(
                x=x,
                metric_per_pipeline=metric_per_pipeline,
                max_num_pipelines=max_num_pipelines,
            )
        except ValueError as e:
            raise e

        lcb = mean - torch.sqrt(self.beta) * stddev
        # with each round/batch of evaluation the exploration factor is reduced
        self.t += 1
        self.beta *= self.beta_decay**self.t

        return lcb

    def set_state(self, surrogate_model, incumbent, **kwargs):
        """
        Set the state of the acquisition function.

        Args:
            surrogate_model: the surrogate model to use for the acquisition function.
        """
        super().set_state(surrogate_model)
        self.incumbent = incumbent
