#!/bin/bash

# Define the path to the phem submodule
PHEM_DIR="RegularizedNeuralEnsemblers/posthoc/phem/phem"

# Check if the phem directory already exists
if [ ! -d "$PHEM_DIR" ]; then
    # Clone the repository if it does not exist
    git clone https://github.com/LennartPurucker/phem.git $PHEM_DIR
fi

# Install phem in editable mode using pip
pip install -e $PHEM_DIR

echo "phem has been installed."
