#based on https://github.com/LennartPurucker/phem
#install phem if you want to use this: pip install git+https://github.com/LennartPurucker/phem
from __future__ import annotations

from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
import torch

from phem.methods.ensemble_weighting import CMAES
from phem.base_utils.metrics import make_metric

from ..metadatasets.base_metadataset import BaseMetaDataset
from .base_ensembler import BaseEnsembler

# -- Obtain Base Models from Sklearn

def gather_predictions(models, X):
    predictions = []
    for model in models:
        predictions.append(model.predict_proba(X))

    return predictions




class CMAESEnsembler(BaseEnsembler):
    "Based on CMA-ES for Post Hoc Ensembling in AutoML: A Great Success and Salvageable Failure."

    def __init__(
        self,
        metadataset: BaseMetaDataset | None = None,
        metric_name: str = "error",
        n_iterations: int = 50,
        normalize_weights: str = "softmax",
        trim_weights: str = "ges-like",
        device: torch.device = torch.device("cpu"),
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(metadataset=metadataset, device=device)
        self.metric_name = metric_name
        self.n_iterations = n_iterations
        self.normalize_weights = normalize_weights
        self.trim_weights = trim_weights
        self.random_state = random_state
        self.fitted_models = [None]

        if self.metadataset is not None:
            self.metric_name = metadataset.metric_name
        else:
            #dummy metadataset
            self.metadataset = BaseMetaDataset("")
            self.metric_name = metric_name

        if self.metric_name == "accuracy" or self.metric_name == "error":
            self.metric = make_metric(metric_func=accuracy_score,
                        metric_name="accuracy",
                        maximize=True,
                        classification=True,
                        always_transform_conf_to_pred=True,
                        optimum_value=1)
        elif self.metric_name == "neg_roc_auc":
            self.metric = make_metric(metric_func=roc_auc_score,
                        metric_name="roc_auc_score",
                        maximize=True,
                        classification=True,
                        always_transform_conf_to_pred=True,
                        optimum_value=1) 
        elif self.metric_name == "nll":
            self.metric = make_metric(metric_func=log_loss,
                        metric_name="log_loss",
                        maximize=False,
                        classification=True,
                        always_transform_conf_to_pred=True,
                        optimum_value=0)         
        elif self.metric_name == "mse":
            self.metric = make_metric(metric_func=mean_squared_error,
                        metric_name="mean_squared_error",
                        maximize=False,
                        classification=False,
                        always_transform_conf_to_pred=True,
                        optimum_value=0)              
        else:
            raise NotImplementedError
        self.cmaes = None

    def get_weights(self, *args, **kwargs):

        num_samples = self.metadataset.get_num_samples()
        num_classes = self.metadataset.get_num_classes()
        weights = torch.FloatTensor(self.cmaes.weights_).unsqueeze(0) # change when omitting the batch size

        weights = torch.repeat_interleave(weights.unsqueeze(-1), num_samples, dim=-1)
        weights = torch.repeat_interleave(weights.unsqueeze(-1), num_classes, dim=-1)

        return weights

    def sample(
        self,
        X_obs,
        **kwargs,
    ) -> tuple[list, float]:
        
        self.X_obs = X_obs

        self.cmaes = CMAES(
            base_models=X_obs,
            score_metric=self.metric,
            n_iterations=self.n_iterations,
            normalize_weights=self.normalize_weights,
            trim_weights=self.trim_weights,
            # If the ensemble requires the metric, we assume the labels to be encoded
            random_state=self.random_state,
        )

        predictions = []
        for model_id in X_obs:
            model_prediction = self.metadataset.get_predictions([[model_id]])[0][0].numpy()
            predictions.append(model_prediction)

        labels = self.metadataset.get_targets().numpy()
        self.cmaes.ensemble_fit(predictions=predictions,
                                labels=labels)
        weights = self.get_weights()
        _, metric, metric_per_pipeline, _  = self.metadataset.evaluate_ensembles_with_weights([X_obs], weights)
        self.best_ensemble = X_obs
        return self.best_ensemble, metric
