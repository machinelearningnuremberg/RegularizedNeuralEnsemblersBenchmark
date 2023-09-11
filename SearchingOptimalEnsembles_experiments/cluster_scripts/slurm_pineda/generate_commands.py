import random
config_space = {"beta": [0., 0.01, 0.1, 1],
                "score_with_rank": [0, 1],
                "activation_output": ["sigmoid", "relu"],
                "criterion_type": ["weighted_listwise", "pointwise"],
                "num_inner_epochs": [100, 500, 1000],
                "acquisition_name": ["ei", "lcb"]}

group_id = "DRE01"
conf_list = []
base_command = "--surrogate_name dre --max_num_pipelines 5"

for i in range(50):
    #sample
    experiment_id = "dre_" + str(i)
    beta = random.choice(config_space["beta"])
    score_with_rank = random.choice(config_space["score_with_rank"])
    activation_output = random.choice(config_space["activation_output"])
    criterion_type = random.choice(config_space["criterion_type"])
    num_inner_epochs = random.choice(config_space["num_inner_epochs"])
    acquisition_name = random.choice(config_space["acquisition_name"])

    command = base_command
    command += " --beta " + str(beta)
    command += " --score_with_rank " + str(score_with_rank)
    command += " --activation_output " + str(activation_output)
    command += " --criterion_type " + str(criterion_type)
    command += " --num_inner_epochs " + str(num_inner_epochs)
    command += " --acquisition_name " + str(acquisition_name)

    for i in range(6):
        temp_command = command
        temp_command += " --dataset_id " + str(i)
        temp_command += " --experiment_id " + experiment_id
        temp_command += " --group_id " + group_id
        conf_list.append(temp_command)

#write conf list to file
with open("bash_args/DRE01.args", "w") as f:
    for conf in conf_list:
        f.write(conf + "\n")