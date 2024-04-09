# type: ignore
# pylint: skip-file

import itertools
import random

import random
import itertools
config_space = {
    "beta": [0.0, 0.01, 0.1, 1],
    "score_with_rank": [0, 1],
    "activation_output": ["sigmoid", "relu"],
    "criterion_type": ["weighted_listwise", "pointwise"],
    "num_inner_epochs": [100, 500, 1000],
    "acquisition_name": ["ei", "lcb"],
}

config_space = {"max_num_pipelines": [1, 2, 4, 6, 8, 10]}

config_space = {
    "max_num_pipelines": [1, 2],
    "num_estimators": [10, 100, 500],
    "beta": [0.0, 0.01, 0.1],
}

#config_space = {"max_num_pipelines": [1, 6], "metadataset": ["scikit-learn"]}


#config_space = {"max_num_pipelines": [1, 6]}

config_space = {
    "hidden_dim": [ 128],
    "num_layers_ff": [3],
    "num_heads": [4],
    "lr": [ 0.00001],
    "num_inner_epochs": [100, 1000],
    "metric_name": ["error", "nll"],
    "meta_num_epochs": [0, 1000],
    "initial_design_size" : [1]
}
#config_space = {"max_num_pipelines": [5],
#                "metric_name": ["error"]
#                }

#config_space = {"max_num_pipelines": [5, 6]}

# base_command = "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output sigmoid" \
#    " --criterion_type weighted_listwise --num_inner_epochs 100 --acquisition_name lcb"
# )
#base_command = "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output sigmoid" \
#               " --criterion_type weighted_listwise --num_inner_epochs 1000 --acquisition_name lcb" \
#               " --log-wandb"
base_command = (
    "--surrogate_name dre --beta 0.1 --score_with_rank 0 --activation_output relu --criterion_type pointwise"
    " --acquisition_name ei"
)
# base_command = "--searcher_name random"

# base_command = "--searcher_name leo --surrogate_name rf"
# base_command = "--searcher_name divbo --surrogate_name rf"
# batch_id = "LEO02"
# batch_id = "DIVBO01"
# batch_id = "RS01"

# base_command = "--searcher_name bo --surrogate_name dkl"
# batch_id = "DKL01"

experiments = {
    "DRE13": "--surrogate_name dre --beta 0.1 --score_with_rank 0 --activation_output relu --criterion_type pointwise"
    " --num_inner_epochs 500 --acquisition_name ei",
    "DRE14": "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output sigmoid --criterion_type weighted_listwise --num_inner_epochs 100 --acquisition_name lcb",
    "DKL02": "--searcher_name bo --surrogate_name dkl",
    "RS03": "--searcher_name random",
    "LEO03": "--searcher_name leo --surrogate_name rf",
    "DIVBO02": "--searcher_name divbo --surrogate_name rf",
}

experiments = {
    "DKL03": "--searcher_name bo --surrogate_name dkl --hidden_dim 128 --lr 0.0001 --metadataset quicktune",
    "DKL04": "--searcher_name bo --surrogate_name dkl --hidden_dim 128 --lr 0.0001 --metadataset scikit-learn",
}

experiments = {
    "RS04": "--searcher_name random --metadataset quicktune --no_posthoc",
    "RS05": "--searcher_name random --metadataset scikit-learn --no_posthoc",
}

experiments = {
    "DRE17": "--surrogate_name dre --beta 0.1 --score_with_rank 0 --activation_output relu --criterion_type pointwise --max_num_pipelines 5",
    "DRE18": "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output sigmoid --criterion_type weighted_listwise --max_num_pipelines 5",
}

experiments = {"DIVBO03": "--searcher_name divbo --surrogate_name rf"}

#--surrogate_name dre --beta 0.1 --score_with_rank 0 --activation_output relu --criterion_type pointwise --max_num_pipelines 5 --dataset_id 0 --run_name DRE15_0_0 --seed 0 --hidden_dim 64 --num_layers_ff 3 --num_heads 8 --lr 0.0001 --experiment_group DRE15_7

experiments = {"DRE24": "--surrogate_name dre --beta 0.1 --score_with_rank 0 --activation_output relu --criterion_type pointwise --max_num_pipelines 5",
               "DRE25": "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output sigmoid --criterion_type weighted_listwise --max_num_pipelines 5"}
#experiments = {"DIVBO05": "--searcher_name divbo --surrogate_name rf"}


num_experiments = 1
num_seeds = 3
counter = 0
num_datasets = 6

for batch_id, base_command in experiments.items():
    conf_list = []
    catersian_product = list(itertools.product(*config_space.values()))
    for i in range(num_seeds):
        for j in range(num_datasets):
            experiment_id = batch_id + "_" + str(j) + "_" + str(i)

            temp_command = base_command
            temp_command += " --dataset_id " + str(j)
            temp_command += " --run_name " + experiment_id
            temp_command += " --seed " + str(i)

            # cartesian product of lists with itertools
            for k, combination in enumerate(catersian_product):
                temp_command2 = temp_command
                # value = random.choice(config_space[key])
                for arg, value in zip(config_space.keys(), combination):
                    temp_command2 += " --" + arg + " " + str(value)
                temp_command2 += " --experiment_group " + batch_id + "_" + str(k)
                conf_list.append(temp_command2)

    # write conf list to file
    with open(f"bash_args/{batch_id}.args", "w") as f:
        for conf in conf_list:
            f.write(conf + "\n")
