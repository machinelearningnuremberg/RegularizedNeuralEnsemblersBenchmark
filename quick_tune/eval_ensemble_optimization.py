"""
Implementation of the ensemble optimization algorithm. Paper: https://arxiv.org/pdf/1905.06159.pdf
Author:
"""

import numpy as np
import pandas as pd
import torch
from metadataset import EnsembleMetaDataset
from surrogates import create_surrogate
from scipy.stats import norm as norm
import warnings
from typing import List, Optional, Tuple
import os
from sklearn.exceptions import ConvergenceWarning
import json
import argparse

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class EnsembleOptimization:

    def __init__(self, metadataset, ensemble_size: int = 10,
                                    initialization_method: str = "randomly",
                                    num_iterations: int = 200,
                                    surrogate_type: str = "GP",
                                    use_oracle: bool = False,
                                    device: str = "cuda"):

        self.ensemble = None
        self.device = device
        self.ensemble_size = ensemble_size
        self.metadataset = metadataset
        self.use_oracle = use_oracle
        self.num_iterations = num_iterations
        self.initialization_method = initialization_method
        self._build_surrogate(surrogate_type = surrogate_type)
        #self._initialize_ensemble(how = initialization_method)
        #self.observed_pipeline_ids = self.ensemble.copy()


    def init_for_dataset(self, dataset_name):
        self.dataset_name = dataset_name
        self.metadataset.set_dataset(dataset_name, split="val")
        self.hp_candidates = np.array(self.metadataset.get_hp_candidates().values)
        self.input_dime = self.hp_candidates.shape[1]
        self.num_candidates = self.hp_candidates.shape[0]
        self._initialize_ensemble(how = self.initialization_method)
        self.observed_pipeline_ids = self.ensemble.copy()

    def _build_surrogate(self, surrogate_type: str = "GP"):
        """Builds the surrogate model.
        Args:
            - surrogate_type: str, the type of surrogate model to use.
        """

        if surrogate_type == "GP":
            self.surrogate = create_surrogate("GP")
        elif surrogate_type == "RF":
            self.surrogate = create_surrogate("RFSK", n_estimators=100)
        else:
            raise NotImplementedError



    def _initialize_ensemble(self, how: str ="randomly"):

        """Initializes the ensemble.
        Args:
            - how: str, the method to initialize the ensemble.
        """

        if how == "randomly":
            self.ensemble = np.random.randint(0, self.num_candidates, self.ensemble_size).tolist()

        else:
            raise NotImplementedError


    def _get_observations(self):
        """Returns the candidates to fit the surrogate model.
        Returns:
            X: All the *observed* pipelines candidates.
            y: The performance after adding the pipeline to the ensemble."""

        ensemble_ids = np.array(self.ensemble).reshape(1,-1).repeat(len(self.observed_pipeline_ids),0)
        observed_ids = np.array(self.observed_pipeline_ids).reshape(-1,1)
        ids = np.concatenate((ensemble_ids, observed_ids), axis=1)
        pipeline_hps, y, _ = self.metadataset.get_y_ensemble(ids, batch_size=ids.shape[0])
        X = pipeline_hps[:,-1,:].cpu().numpy()
        y = y.cpu().numpy()
        self.best_f = np.max(y)
        return X, y

    def EI(self, mean, sigma, best_f, epsilon=0):
        with np.errstate(divide='warn'):
            imp = mean - best_f - epsilon
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _get_next_pipeline_to_observe(self) -> np.array:

        """Returns the next pipeline to observe.
        The candidate is selected from the observed pipelines.
        Returns: int, the id of the next pipeline to observe."""

        mean, sigma = self.surrogate.predict(self.hp_candidates)

        acq_value = self.EI(mean, sigma, self.best_f)

        return np.argmax(acq_value)


    def _aqcf(self, X) -> np.array:

        return None

    def _get_normalized_regret(self, y)-> float:

        best_perf = self.metadataset.get_best_performance()
        worst_perf = self.metadataset.get_worst_performance()
        normalized_regret = (best_perf - y)/(best_perf - worst_perf)
        return normalized_regret.item()

    def _get_next_pipeline_to_add(self) -> Tuple[int, float]:

        """Returns the next pipeline to add to the ensemble.
        The candidate is selected from the observed pipelines."""

        if self.use_oracle:

            return self.metadataset.get_best_performance_idx().item(), self.metadataset.get_best_performance().item()
        else:
            ensemble_ids = np.array(self.ensemble).reshape(1,-1).repeat(len(self.observed_pipeline_ids),0)
            observed_ids = np.array(self.observed_pipeline_ids).reshape(-1,1)
            ids = np.concatenate((ensemble_ids, observed_ids), axis=1)
            pipeline_hps, y, _ = self.metadataset.get_y_ensemble(ids, batch_size=ids.shape[0])
            y = y.cpu().numpy()
            argmax_set = np.where(y == np.max(y))[0]
            np.random.shuffle(argmax_set)
            pipeline_id = self.observed_pipeline_ids[argmax_set[0]]
            return pipeline_id, y[argmax_set[0]].item()

    def test_best_ensemble_on_dataset(self, ensemble = None):

        if ensemble is None:
            ensemble = self.best_ensemble

        self.metadataset.set_dataset(self.dataset_name, split = "test")
        _, y_ensemble, _ = self.metadataset.get_y_ensemble(ensemble)
        best_perf = self.metadataset.get_best_performance()
        worst_perf = self.metadataset.get_worst_performance()
        normalized_regret = (best_perf - y_ensemble.item())/(best_perf - worst_perf)
        self.metadataset.set_dataset(self.dataset_name, split = "val")

        return normalized_regret.item()


    def post_hoc_ensembling(self, initial_ensemble, ensemble_size = 5):

        post_hoc_ensemble = initial_ensemble.copy()
        runtime = 0
        for i in range(ensemble_size-1):
            next_to_add = None
            best_y = -np.inf

            for j in range(len(self.observed_pipeline_ids)):
                temp_ensemble = post_hoc_ensemble.copy()+ [self.observed_pipeline_ids[j]]
                _, y, _ = self.metadataset.get_y_ensemble(temp_ensemble)
                if y > best_y:
                    best_y = y
                    next_to_add = self.observed_pipeline_ids[j]

            post_hoc_ensemble.append(next_to_add)
        _, post_hoc_y, _ = self.metadataset.get_y_ensemble(post_hoc_ensemble)
        post_hoc_regret_val = self._get_normalized_regret(post_hoc_y)
        post_hoc_regret_test = self.test_best_ensemble_on_dataset(ensemble = post_hoc_ensemble)
        return post_hoc_regret_val, post_hoc_regret_test, runtime, post_hoc_ensemble

    def random_ensemble(self, ensemble_size = 5, replace=True):

        ensemble = np.random.choice(self.observed_pipeline_ids, ensemble_size, replace=replace).tolist()
        _, random_y, _ = self.metadataset.get_y_ensemble(ensemble)
        randon_regret_val = self._get_normalized_regret(random_y)
        random_regret_test = self.test_best_ensemble_on_dataset(ensemble = ensemble)
        return randon_regret_val, random_regret_test, runtime, ensemble

    def optimize(self):

        val_performance_curve = []
        test_performance_curve = []
        time_curve = []
        best_norm_regret = 1

        for i in range(self.num_iterations):

            j = i % self.ensemble_size
            self.ensemble.pop(j)
            X, y = self._get_observations()

            #fit surrogate
            self.surrogate.fit(X, y)
            next_to_observe = self._get_next_pipeline_to_observe()

            if next_to_observe not in self.ensemble:
                self.observed_pipeline_ids.append(next_to_observe)

            next_to_add, perf = self._get_next_pipeline_to_add()
            self.ensemble.insert(j, next_to_add)
            norm_regret = self._get_normalized_regret(perf)

            if norm_regret < best_norm_regret:
                best_norm_regret = norm_regret
                self.best_ensemble = self.ensemble.copy()

            val_performance_curve.append(best_norm_regret)
            test_performance_curve.append(self.test_best_ensemble_on_dataset())
            print("Iteration: {}, Best performance: {}".format(i, np.min(val_performance_curve)))

        return val_performance_curve, test_performance_curve, time_curve, self.ensemble

    def save_results(self, val_curve, test_curve, time_curve, best_ensemble, experiment_id):
        results_dict = {"val_perf": val_curve,
                        "test_perf": test_curve,
                        "runtime": time_curve,
                        "best_ensemble": best_ensemble,
                        }
        save_path = os.path.join(current_dir, "..", "results", experiment_id, self.dataset_name)
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "results.json"), "w") as f:
            json.dump(str(results_dict), f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_size', type=int, default=5)
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--experiment_id', type=str, default="EO_oracle")
    parser.add_argument('--optimization_type', type=str, default="", choices=["", "oracle", "post_hoc", "random"])
    parser.add_argument('--surrogate_type', type=str, default="gp", choices=["GP", "RF"])

    args = parser.parse_args()

    ensemble_size = args.ensemble_size
    num_iterations = args.num_iterations
    experiment_id = args.experiment_id
    optimization_type = args.optimization_type
    surrogate_type = args.surrogate_type


    if optimization_type == "oracle":
        use_oracle = True
    else:
        use_oracle = False

    # dataset = loader.datasets[0]
    current_dir = os.path.dirname(os.path.realpath(__file__))

    #max_num_pipelines is not useful here
    metadataset = EnsembleMetaDataset( data_path= os.path.join(current_dir, "..", "..", "AutoFinetune",
                                                                  "aft_data" , "predictions"),)

    for dataset in metadataset.datasets:
        if optimization_type in  ["post_hoc", "random", "oracle"]:
            temp_ensemble_size = 1
        else:
            temp_ensemble_size = ensemble_size

        eo = EnsembleOptimization(metadataset=metadataset,
                                  ensemble_size=temp_ensemble_size,
                                  num_iterations=num_iterations,
                                  use_oracle=use_oracle,
                                  surrogate_type=surrogate_type)
        eo.init_for_dataset(dataset)
        val_perf, test_perf, runtime, best_ensemble = eo.optimize()

        if optimization_type == "post_hoc":
            val_perf_post, test_perf_post, \
            runtime_post, best_ensemble_post = eo.post_hoc_ensembling(eo.best_ensemble, ensemble_size=ensemble_size)
            eo.save_results(val_perf_post, test_perf_post, runtime_post, best_ensemble_post, experiment_id+"_post")
        elif optimization_type == "random":
            val_perf_random, test_perf_random, \
            runtime_random, best_ensemble_random = eo.random_ensemble(ensemble_size=ensemble_size)
            eo.save_results(val_perf_random, test_perf_random, runtime_random, best_ensemble_random, experiment_id+"_random")

        eo.save_results(val_perf, test_perf, runtime, best_ensemble, experiment_id)


