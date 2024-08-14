# type: ignore
# pylint: skip-file
import itertools
import os
import random
import warnings

import yaml
import argparse

warnings.filterwarnings("ignore", category=UserWarning, message="[LightGBM]")

parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, default="SOE")
args = parser.parse_args()

project_name = args.project_name

with open(f"experiments_confs/{project_name}.yml") as f:
    experiments_conf = yaml.safe_load(f)


def extend_command(command, args):
    for arg_name, arg_value in args:
        command += f" --{arg_name} {arg_value}"
    return command


num_datasets = experiments_conf.pop("num_datasets")
num_meta_splits = experiments_conf.pop("num_meta_splits")
experiments = experiments_conf.pop("experiments")
project_name = experiments_conf["project_name"]
num_seeds = experiments_conf.pop("num_seeds", 1)
counter = 0

for experiment_group, experiment in experiments.items():
    conf_list = []
    base_command = experiment.pop("base_command")
    hyper_grid = experiment.pop("hyper_grid")
    base_command = extend_command(base_command, experiments_conf.items())
    catersian_product = list(itertools.product(*hyper_grid.values()))

    for i in range(num_meta_splits):
        for j in range(num_datasets):
            experiment_id = experiment_group + "_" + str(j) + "_" + str(i)
            temp_command = base_command
            temp_command += " --dataset_id " + str(j)
            temp_command += " --run_name " + experiment_id
            temp_command += " --meta_split_id " + str(i)

            # cartesian product of lists with itertools
            for k, combination in enumerate(catersian_product):
                temp_args = zip(hyper_grid.keys(), combination)
                temp_command2 = extend_command(temp_command, temp_args)

                for l in range(num_seeds):
                    temp_command3 = temp_command2 + " --experiment_group " + experiment_group + "_" + str(k) + f" --seed {l}"
                    conf_list.append(temp_command3)

    # write conf list to file
    os.makedirs(f"bash_args/{project_name}", exist_ok=True)
    with open(f"bash_args/{project_name}/{experiment_group}.args", "w") as f:
        for conf in conf_list:
            f.write(conf + "\n")
