import logging

import wandb

import SearchingOptimalEnsembles as SOE

logging.basicConfig(level=logging.DEBUG)

# metatadaset_name = "quicktune"
metatadaset_name = "scikit-learn"
surrogate_name = "dkl"


try:
    wandb.init(
        project="SearchingOptimalEnsembles", group=f"{metatadaset_name}_{surrogate_name}"
    )
except wandb.errors.UsageError:
    print("Wandb is not available")

SOE.run(metadataset_name=metatadaset_name, surrogate_name=surrogate_name)
