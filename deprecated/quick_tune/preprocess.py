# type: ignore
# pylint: skip-file

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["model", "opt", "opt_betas", "sched", "dataset"]
COLS_TO_DROP = ["project_name", "device", "project"]
NON_CAT_COLS = [
    "batch_size",
    "bss_reg",
    "cotuning_reg",
    "cutmix",
    "decay_epochs",
    "decay_rate",
    "delta_reg",
    "drop",
    "linear_probing",
    "lr",
    "mixup",
    "mixup_prob",
    "momentum",
    "patience_epochs",
    "pct_to_freeze",
    "ra_magnitude",
    "ra_num_ops",
    "random_augment",
    "smoothing",
    "sp_reg",
    "stoch_norm",
    "trivial_augment",
    "warmup_epochs",
    "warmup_lr",
    "weight_decay",
    "clip_grad",
    "layer_decay",
]


def preprocess_args(
    args_df,
    drop_constant_args=False,
    encode_categorical_args=True,
    impute_numerical_args=True,
):
    index = args_df.index
    args_df["opt_betas"] = args_df["opt_betas"].apply(lambda x: str(x))
    if drop_constant_args:
        constant_cols = []
        for col in list(args_df.columns):
            if len(args_df[col].astype(str).unique()) == 1:
                constant_cols.append(col)

        for column in constant_cols + COLS_TO_DROP:
            if column in args_df.columns:
                args_df = args_df.drop(columns=column)

    # cat_columns = args_df.select_dtypes(include=['object']).columns.tolist()
    cat_columns = CAT_COLS
    if encode_categorical_args:
        cat_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        col_transformer = ColumnTransformer(
            transformers=[("cat", cat_transformer, cat_columns)]
        )

        # Fit and transform the data
        df_cat_transformed = col_transformer.fit_transform(args_df).toarray()
        new_cat_columns = col_transformer.get_feature_names_out().tolist()
        df_cat_transformed = pd.DataFrame(df_cat_transformed, columns=new_cat_columns)
    else:
        df_cat_transformed = args_df.select_dtypes(include=["object"])

    # non_cat_columns = args_df.select_dtypes(include=['float64', 'int64', 'bool']).columns.tolist()
    non_cat_columns = NON_CAT_COLS
    args_df["clip_grad"][args_df["clip_grad"] == "None"] = np.nan
    args_df["layer_decay"][args_df["layer_decay"] == "None"] = np.nan

    if impute_numerical_args:
        imputer = SimpleImputer(strategy="constant", fill_value=-1)
        df_non_cat_transformed = imputer.fit_transform(args_df[non_cat_columns])
        df_non_cat_transformed = pd.DataFrame(
            df_non_cat_transformed, columns=non_cat_columns
        )
    else:
        df_non_cat_transformed = args_df.select_dtypes(
            include=["float64", "int64", "bool"]
        )

    col_names = df_non_cat_transformed.columns.tolist()
    df_non_cat_transformed = StandardScaler().fit_transform(df_non_cat_transformed.values)
    df_non_cat_transformed = pd.DataFrame(df_non_cat_transformed, columns=col_names)
    args_df = pd.concat([df_cat_transformed, df_non_cat_transformed], axis=1)

    args_df.index = index

    return args_df


KEEP_COLUMNS = [
    "batch_size",
    "bss_reg",
    "cotuning_reg",
    "cutmix",
    "random_augment",
    "delta_reg",
    "drop",
    "lr",
    "mixup",
    "mixup_prob",
    "ra_num_ops",
    "ra_magnitude",
    "patience_epochs",
    "opt_betas",
    "weight_decay",
    "warmup_lr",
    "warmup_epochs",
    "stoch_norm",
    "opt",
    "sched",
    "trivial_augment",
    "layer_decay",
    "auto_augment",
    "linear_probing",
    "clip_grad",
    "smoothing",
    "pct_to_freeze",
    "sp_reg",
    "model",
    "momentum",
    "decay_epochs",
    "decay_rate",
    "dataset",
]


if __name__ == "__main__":
    aggregated_args_path = "../AutoFinetune/aft_data/predictions/aggregated_args.json"
    with open(aggregated_args_path) as f:
        args = json.load(f)

    # gather the args
    # args = []
    # for dataset_name in aggregated_info.keys():
    #    X = aggregated_info[dataset_name]["args"]
    #    args += X

    # save json
    # with open("../AutoFinetune/aft_data/predictions/preprocessed_args.json", "w") as f:
    #    json.dump(args, f)
    df_args = pd.DataFrame(args)[KEEP_COLUMNS]
    df_args["dataset"] = df_args["dataset"].apply(lambda x: x.replace("/", "_"))
    preprocessed_df_args = preprocess_args(df_args)
    preprocessed_df_args.to_csv(
        "../AutoFinetune/aft_data/predictions/preprocessed_args.csv", index=False
    )
