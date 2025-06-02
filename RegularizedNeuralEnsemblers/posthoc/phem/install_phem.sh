#!/bin/bash

# Define the path to the phem submodule
PHEM_DIR="RegularizedNeuralEnsemblers/posthoc/phem/phem"

# Install phem in editable mode using pip
pip install -e $PHEM_DIR

echo "phem has been installed."
