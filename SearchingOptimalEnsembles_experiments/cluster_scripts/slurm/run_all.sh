#!/bin/bash

basedir=/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/
workdir=/work/dlclarge2/janowski-quicktune/SearchingOptimalEnsembles/SearchingOptimalEnsembles_experiments/

python "$basedir"/cluster_scripts/slurm/generate_cmds.py

python "$basedir"cluster_scripts/slurm/slurm_helper.py \
    -q alldlc_gpu-rtx2080 \
    --timelimit 86400 \
    --startup "$basedir"cluster_scripts/slurm/startup.sh \
    --array_min 1 \
    --array_max 150 \
    --memory_per_job "30000mb" \
    -o "$workdir"LOGS/ \
    -l "$workdir"LOGS/ \
    "$basedir"cluster_scripts/slurm/ablate_DKL_1.txt
