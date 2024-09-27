import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
import copy

from .base import BasePipelineSampler

param_distributions_classification = {
    RandomForestClassifier: {
        'n_estimators': randint(10, 100),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 11)
    },
    GradientBoostingClassifier: {
        'n_estimators': randint(10, 100),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(3, 10)
    },
    SVC: {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.01, 1),
        'kernel': ['linear', 'rbf', 'poly'],
        'max_iter': [1000],
        'probability': [True]
    },
    LogisticRegression: {
        'C': uniform(0.1, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
}

param_distributions_regression = {
    RandomForestRegressor: {
        'n_estimators': randint(10, 100),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 11)
    }
}

class RandomPipelineSampler(BasePipelineSampler):

    def __init__(self, num_pipelines: int = 20, random_state: int = 42,
                task_type="classification"):
        self.num_pipelines = num_pipelines
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)
        self.task_type = task_type

        if task_type == "classification":
            self.param_distributions = param_distributions_classification
        elif task_type == "regression":
            self.param_distributions = param_distributions_regression
        else:
            raise ValueError("No valid task type.")

    def sample(self):
        pipelines = []
        hps = []
        models = []
        for i in range(self.num_pipelines):
            model_class_id = self.rng.choice(len(self.param_distributions.keys()))
            model_class = copy.deepcopy(list(self.param_distributions.keys())[model_class_id])
            parameters= list(ParameterSampler(param_distributions=self.param_distributions[model_class], 
                                                n_iter=1, 
                                                random_state=self.random_state+i))[0]
            
            pipelines.append(model_class(**parameters))
            hps.append(parameters)
            models.append(str(model_class))
        
        hps = pd.DataFrame(hps)
        hps["model"] = models
        hps = self.process_hps(hps)
        return pipelines, hps
    

if __name__ == "__main__":
    sampler = RandomPipelineSampler()
    print(sampler.sample())