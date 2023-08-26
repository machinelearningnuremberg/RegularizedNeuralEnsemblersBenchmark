from functools import wraps

import numpy as np
import torch
from scipy.optimize import differential_evolution
from scipy.stats import norm


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
