# pylint: skip-file

import numpy as np


class DatasetGenerator:
    def __init__(
        self,
        num_base_functions=2,
        lower_limit=-10,
        upper_limit=10,
        num_initial_observed_params=100,
        add_noise=False,
        dataset_size=1000,
        search_space_size=100,
        simple_coefficients=False,
        sample_amplitude=True,
    ):
        self.num_base_functions = num_base_functions
        self.x = np.arange(
            lower_limit, upper_limit, (upper_limit - lower_limit) / dataset_size
        )
        self.theta_grid = np.arange(1, upper_limit, (upper_limit - 1) / search_space_size)
        self.simple_coffients = simple_coefficients
        self.target_dict = dict(
            zip([i for i in self.theta_grid], np.arange(len(self.theta_grid)))
        )
        self.add_noise = add_noise
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.num_initial_observed_params = num_initial_observed_params
        self.sample_amplitude = sample_amplitude
        self.random_generator = None
        self.base_functions = self.get_base_functions(self.x)

    def get_base_functions(self, x):
        base_observations = np.array(
            [(np.sin(2 * np.pi * x / theta) + 1) for theta in self.theta_grid]
        )
        return base_observations

    def get_functions(self, params):
        assert self.random_generator is not None, "Random generator cannot be None"
        y_base = []
        w_base = []
        for a, theta in params:
            rho = self.random_generator.uniform(self.upper_limit / 2, self.upper_limit)
            if self.simple_coffients:
                w = a * np.ones(self.x.shape)
            else:
                w = a * (np.cos(2 * np.pi * self.x / rho) + 1)
            y_base.append((np.sin(2 * np.pi * self.x / theta) + 1) / 100)
            w_base.append(w)

        y_base = np.array(y_base)
        w_base = np.array(w_base)
        y = np.divide(y_base * w_base, w_base.sum(axis=0, keepdims=1)).sum(0).reshape(-1)
        return y, y_base, w_base

    def get_data(self, random_generator_seed=10):
        self.random_generator = np.random.default_rng(random_generator_seed)
        params = []
        for i in range(self.num_base_functions):
            if self.sample_amplitude:
                a = self.random_generator.uniform(1, self.upper_limit)
            else:
                a = 1
            theta = self.theta_grid[
                self.random_generator.integers(0, len(self.theta_grid))
            ]
            params.append((a, theta))
        y, y_base, w_base = self.get_functions(params)

        if self.add_noise:
            y += self.random_generator.normal(0, 0.1, y.shape)

        true_theta = np.array([self.target_dict[theta] for a, theta in params])
        params = np.array(params)

        return y, y_base, params, self.x, true_theta, w_base, self.base_functions
