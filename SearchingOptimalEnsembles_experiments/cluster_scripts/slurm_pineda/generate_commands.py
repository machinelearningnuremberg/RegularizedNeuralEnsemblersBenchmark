import random
config_space = {"beta": [0., 0.01, 0.1, 1],
                "score_with_rank": [0, 1],
                "activation_output": ["sigmoid", "relu"],
                "criterion_type": ["weighted_listwise", "pointwise"],
                "num_inner_epochs": [100, 500, 1000],
                "acquisition_name": ["ei", "lcb"]}

config_space = {"max_num_pipelines" : [1,2,4,6,8,10]}


conf_list = []

base_command = "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output relu" \
               " --criterion_type weighted_listwise --num_inner_epochs 1000 --acquisition_name lcb" \
               " --log-wandb"
#base_command = "--surrogate_name dre --beta 0.01 --score_with_rank 1 --activation_output sigmoid" \
#               " --criterion_type weighted_listwise --num_inner_epochs 1000 --acquisition_name lcb" \
#               " --log-wandb"
#base_command = "--beta 0.1 --score_with_rank 1 --activation_output relu --criterion_type pointwise" \
#               " --num_inner_epochs 500 --acquisition_name ei --log-wandb"
base_command = "--searcher_name random --log-wandb"

group_id = "RS00"
experiment_prefix = "rs"

num_experiments = 1
counter = 0
for i in range(num_experiments):
    #sample
    for key in config_space.keys():
        #value = random.choice(config_space[key])
        for value in config_space[key]:
            experiment_id = experiment_prefix + "_" + str(counter)
            for j in range(6):
                temp_command = base_command + " --" + key + " " + str(value)
                temp_command += " --dataset_id " + str(j)
                temp_command += " --experiment_id " + experiment_id
                temp_command += " --group_id " + group_id
                conf_list.append(temp_command)
            counter += 1
#write conf list to file
with open(f"bash_args/{group_id}.args", "w") as f:
    for conf in conf_list:
        f.write(conf + "\n")
