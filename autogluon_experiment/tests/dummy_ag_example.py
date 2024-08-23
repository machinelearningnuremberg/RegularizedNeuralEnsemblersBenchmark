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
        hyperparameters = {
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

    default_arguments = {
        "ne_epochs": 10000,
        "ne_dropout_rate": 0.75,
        "ag_args_ensemble": {"refit_folds": True},  # refit as no validation data is used for early stopping
    }
    hyperparameters_stacking[AutoGluonNeuralEnsembler] = [
        {
            "ne_net_mode": "model_averaging",
            "ag_args": {"name_suffix": "ModelAveraging"},
            **default_arguments,
        },
        {"ne_net_mode": "stacking", "ag_args": {"name_suffix": "Stacking"}, **default_arguments},
    ]
    expected_name_ma = "AutoGluonNeuralEnsemblerModelAveraging_BAG_L2"
    expected_name_st = "AutoGluonNeuralEnsemblerStacking_BAG_L2"

    predictor = predictor.fit_extra(
        hyperparameters=hyperparameters_stacking,
        time_limit=None,
        base_model_names=predictor.model_names(level=1, can_infer=True),
        **predictor_kwargs,
    )

    # -- Run Final Weighted Ensemble Combinations
    l1_bms = predictor.model_names(level=1, can_infer=True)
    l2_bms = predictor.model_names(level=2, can_infer=True)
    assert expected_name_st in l2_bms, "Stacking NE model not found in level 2 models!"
    assert expected_name_ma in l2_bms, "Model Averaging NE model not found in level 2 models!"
    predictor.fit_weighted_ensemble(
        base_models=l1_bms + l2_bms,
        name_suffix="_AllModels",
    )
    for bm in l2_bms:
        predictor.fit_weighted_ensemble(
            base_models=[*l1_bms, bm],
            name_suffix=f"_Only{bm}",
        )

    l2_bms.remove(expected_name_ma)
    l2_bms.remove(expected_name_st)
    predictor.fit_weighted_ensemble(
        base_models=l1_bms + l2_bms,
        name_suffix="_AllModelsWithOutNeuralEnsemblers",
    )

    results = predictor.leaderboard(test_data, display=True)
