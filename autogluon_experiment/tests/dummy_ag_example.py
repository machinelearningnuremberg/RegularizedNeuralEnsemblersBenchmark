from __future__ import annotations

from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon_experiment.ag_neural_ensembler import AutoGluonNeuralEnsembler
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split

TEST_RUN = True

for load_func, task_type, eval_metric in [
    (load_iris, "multiclass", "roc_auc_ovo_macro"),
    (load_breast_cancer, "binary", "roc_auc"),
    # will crash for now due to default inference not supporting regression
    (load_diabetes, "regression", "rmse"),
]:
    data, y = load_func(as_frame=True, return_X_y=True)
    data["target"] = y
    is_classification = task_type in ["binary", "multiclass"]
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=y if is_classification else None,
    )

    predictor_kwargs = dict(
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        num_cpus=8,
        num_gpus=0,
        memory_limit=10,
        fit_weighted_ensemble=False,
    )

    # -- Run Base Models
    if TEST_RUN:
        hyperparameters = {  # FIXME: remove after testing
            "KNN": [
                {"weights": "uniform", "ag_args": {"name_suffix": "Unif"}},
                {"weights": "distance", "ag_args": {"name_suffix": "Dist"}},
            ],
        }

    else:
        hyperparameters = None
    predictor = TabularPredictor(
        label="target",
        problem_type=task_type,
        eval_metric=eval_metric,
        verbosity=2,
    ).fit(
        train_data,
        time_limit=None if TEST_RUN else 60,
        hyperparameters=hyperparameters,
        presets="best_quality",
        num_stack_levels=0,
        **predictor_kwargs,
    )

    # -- Run Stacking Models
    if TEST_RUN:
        hyperparameters_stacking = {}
    else:
        hyperparameters_stacking = get_hyperparameter_config("default")
        del hyperparameters_stacking["KNN"]  # not usable for stacking

    hyperparameters_stacking[AutoGluonNeuralEnsembler] = [{}]  # only default config

    predictor = predictor.fit_extra(
        hyperparameters=hyperparameters_stacking,
        time_limit=None,
        base_model_names=predictor.model_names(level=1, can_infer=True),
        **predictor_kwargs,
    )

    results = predictor.leaderboard(test_data, display=True)
