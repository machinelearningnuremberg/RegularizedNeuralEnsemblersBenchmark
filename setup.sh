#!/bin/bash

# Set strict mode for bash scripting: stop on error, unset variable use, and pipeline errors
set -euo pipefail

# Function to display usage information
usage() {
    echo "Usage: $0 [install-pipeline_bench]"
    exit 1
}

# Check for the correct number of arguments (allow 0 or 1 arguments)
if [ "$#" -gt 1 ]; then
    usage
fi

# Get the argument that was passed to this script, default to an empty string if none
arg=${1-""}

# Function to check Conda commands
check_conda() {
    if ! type conda > /dev/null 2>&1; then
        echo "Conda is not installed. Please install Conda and rerun this script."
        exit 1
    elif ! type conda activate > /dev/null 2>&1; then
        echo "Conda is installed but not initialized in this shell."
        echo "Please run 'conda init $(basename $SHELL)' and restart your shell."
        exit 1
    fi
}

# Function to manage Conda environments and determine python version
manage_conda_env() {


    # Check Python version
    python_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Detected Python version: $python_version"

    # Assert it's 3.10
    if [[ "$python_version" != "3.10" ]]; then
        echo "Unsupported Python version. Please use Python 3.10."
        exit 1
    fi

    # Initialize and update git submodules
    echo "Initializing and updating git submodules..."
    git submodule update --init --recursive

    # Check and remove existing pipeline-bench entries from pyproject.toml
    PYPROJECT_TOML="pyproject.toml"
    if grep -q "pipeline-bench" "$PYPROJECT_TOML"; then
        echo "Removing existing pipeline-bench entry from pyproject.toml..."
        sed -i '/pipeline-bench/d' "$PYPROJECT_TOML"
    fi

    # Install dependencies with Poetry
    echo "Locking and installing dependencies with Poetry..."
    poetry lock
    poetry install

    # Check if the argument is "install-pipeline_bench"
    if [ "$arg" == "install-pipeline_bench" ]; then
        echo "Installation to run experiments with Pipeline-Bench with TPOT search space..."
        # Install pipeline_bench submodule
        echo "Installing pipeline_bench..."
        poetry run install-pipeline_bench || { echo 'Failed to install pipeline_bench submodule'; exit 1; }
    else
        echo "Default installation..."
        echo "Installing tabrepo..."
        poetry run install-tabrepo || { echo 'Failed to install tabrepo submodule'; exit 1; }
        echo "Installing phem..."
        poetry run install-phem || { echo 'Failed to install phem submodule'; exit 1; }
    fi

    poetry install
}


# Check if conda commands are available
#check_conda

# Ensure Conda is initialized and run environment management
if type conda >/dev/null 2>&1; then
    manage_conda_env
else
    echo "Conda is not available on the system or not properly initialized."
    exit 1
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install || { echo 'Failed to install pre-commit hooks'; exit 1; }

echo "Setup completed successfully."
