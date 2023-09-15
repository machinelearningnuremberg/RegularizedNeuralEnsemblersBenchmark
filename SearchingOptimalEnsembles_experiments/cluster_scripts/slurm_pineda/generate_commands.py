# type: ignore
# pylint: skip-file

import itertools
import random

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

conf_list = []

# base_command = (
#    "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output sigmoid"
#    " --criterion_type weighted_listwise --num_inner_epochs 100 --acquisition_name lcb"
#    " --log-wandb"
# )
# base_command = "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output sigmoid" \
#               " --criterion_type weighted_listwise --num_inner_epochs 1000 --acquisition_name lcb" \
#               " --log-wandb"
# base_command = "--beta 0.1 --score_with_rank 1 --activation_output relu --criterion_type pointwise" \
#               " --num_inner_epochs 500 --acquisition_name ei --log-wandb"
# base_command = "--searcher_name random --log-wandb"

base_command = "--searcher_name leo --surrogate_name rf"
batch_id = "LEO01"

num_experiments = 1
num_seeds = 3
counter = 0
num_datasets = 6
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
