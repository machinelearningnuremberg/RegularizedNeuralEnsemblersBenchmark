import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
import copy


param_distributions = {
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

def sample_pipelines(num_pipelines: int =20, 
                    random_state: int =42):

    pipelines = []
    hps = []
    models = []
    for i in range(num_pipelines):
        model_class_id = np.random.choice(len(param_distributions.keys()))
        model_class = copy.deepcopy(list(param_distributions.keys())[model_class_id])
        parameters= list(ParameterSampler(param_distributions=param_distributions[model_class], 
                                            n_iter=1, 
                                            random_state=random_state+i))[0]
        
        pipelines.append(model_class(**parameters))
        hps.append(parameters)
        models.append(str(model_class))
    
    hps = pd.DataFrame(hps)
    hps["model"] = models
    hps = pd.get_dummies(hps).astype(float)
    hps = hps.fillna(0).values
    return pipelines, hps
    

if __name__ == "__main__":
    sample_pipelines()