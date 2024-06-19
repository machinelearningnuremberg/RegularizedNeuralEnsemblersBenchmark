#!/bin/bash

# Define the path where the tabrepo should be cloned
TABREPO_DIR="SearchingOptimalEnsembles/metadatasets/tabrepo/tabrepo"

# Check if the tabrepo directory already exists
if [ ! -d "$TABREPO_DIR" ]; then
    # Clone the repository if it does not exist
    git clone https://github.com/autogluon/tabrepo.git $TABREPO_DIR
fi

# Install tabrepo in editable mode using pip
pip install -e $TABREPO_DIR

echo "tabrepo has been installed."