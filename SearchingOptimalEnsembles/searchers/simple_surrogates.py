import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern


class BaseSurrogate:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        assert self.model is not None, "Model not implemented"
        self.model.fit(X, y)

    def predict(self, X):
        assert self.model is not None, "Model not implemented"
        pred_mean, pred_std = self.model.predict(X)

        return pred_mean, pred_std


class RandomForestWithUncertainty(BaseSurrogate):
    def __init__(self, **args):
        super().__init__()

        self.model = RandomForestRegressor(**args)

    def predict(self, X):
        pred = np.array([tree.predict(X) for tree in self.model]).T
        pred_mean = np.mean(pred, axis=1)
        pred_var = (pred - pred_mean.reshape(-1, 1)) ** 2
        pred_std = np.sqrt(np.mean(pred_var, axis=1))

        return pred_mean, pred_std

class LightGBMEnsemble(BaseSurrogate):
    def __init__(self, **args):
        super().__init__()
        self.num_models = 100
        self.models = [LGBMRegressor(**args) for _ in range(self.num_models)]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        pred = np.array([model.predict(X) for model in self.models]).T
        pred_mean = np.mean(pred, axis=1)
        pred_var = (pred - pred_mean.reshape(-1, 1)) ** 2
        pred_std = np.sqrt(np.mean(pred_var, axis=1))

        return pred_mean, pred_std


class GaussianProcessSurrogate(BaseSurrogate):
    def __init__(self, kernel="matern", **args):
        super().__init__()

        length_scale = args.get("length_scale", 1.0)
        nu = args.get("nu", 1.5)

        if kernel == "rbf":
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0)
        elif kernel == "matern":
            kernel = Matern(length_scale=length_scale, nu=nu)
        else:
            print("Not specified kernel")
            kernel = None

        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=500)

    def predict(self, X):
        pred_mean, pred_std = self.model.predict(X, return_std=True)
        return pred_mean, pred_std


def create_surrogate(model_name, **args):
    if model_name == "gp":
        model = GaussianProcessSurrogate(**args)

    elif model_name == "rf":
        model = RandomForestWithUncertainty(**args)

    elif model_name == "lightgbm":
        model = LightGBMEnsemble(**args)

    else:
        model = None
        print("No surrogate")

    return model
