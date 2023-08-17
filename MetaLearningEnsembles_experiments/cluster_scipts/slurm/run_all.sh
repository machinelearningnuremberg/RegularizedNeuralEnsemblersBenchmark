#!/bin/bash

basedir=/work/dlclarge2/janowski-quicktune/MetaLearningEnsembles/MetaLearningEnsembles_experiments
workdir=/work/dlclarge2/janowski-quicktune/MetaLearningEnsembles/MetaLearningEnsembles_experiments

python "$basedir"/cluster_scripts/slurm/generate_cmds.py

python "$basedir"cluster_scripts/slurm/slurm_helper.py \
    -q relea_gpu-rtx2080 \
    --timelimit 86400 \
    --startup "$basedir"cluster_scripts/slurm/startup.sh \
    --array_min 1 \
    --array_max 10 \
    --memory_per_job 100000 \
    -o "$workdir"LOGS/ \
    -l "$workdir"LOGS/ \
    "$basedir"cluster_scripts/slurm/all_experiments.txt