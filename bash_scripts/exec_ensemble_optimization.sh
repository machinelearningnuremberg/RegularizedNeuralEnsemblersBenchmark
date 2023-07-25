#!/bin/bash
##SBATCH -p alldlc_gpu-rtx2080
#SBATCH -p relea_gpu-rtx2080
#SBATCH --job-name eval_timm_finetune_pipeline
#SBATCH -o /home/pineda/MetaLearningEnsembles/logs/%A-%a.%x.o
#SBATCH -e /home/pineda/MetaLearningEnsembles/logs/%A-%a.%x.e
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00

source /home/pineda/anaconda3/bin/activate autofinetune
export PYTHONPATH="${PYTHONPATH}:${HOME}/MetaLearningEnsembles"
python quick_tune/eval_ensemble_optimization.py