#!/bin/bash

# Set strict mode for bash scripting: stop on error, unset variable use, and pipeline errors
set -euo pipefail

# Function to display usage information
usage() {
    echo "Usage: $0"
    exit 1
}

# Check for the correct number of arguments
if [ "$#" -ne 0 ]; then
    usage
fi

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

    # Modify pyproject.toml based on Python version
    if [[ "$python_version" == "3.9" ]]; then
        sed -i 's/configspace = "0.6.1"/configspace = "0.4.21"/' pyproject.toml
        sed -i 's/scikit-learn = "1.4.0"/scikit-learn = "1.0.2"/' pyproject.toml
    elif [[ "$python_version" == "3.10" ]]; then
        sed -i 's/configspace = "0.4.21"/configspace = "0.6.1"/' pyproject.toml
        sed -i 's/scikit-learn = "1.0.2"/scikit-learn = "1.4.0"/' pyproject.toml
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

    if [[ "$python_version" == "3.10" ]]; then
        echo "Python 3.10 detected, setting up accordingly."
        # Install tabrepo and phem submodules
        echo "Installing pipeline_bench..."
        poetry run install-pipeline_bench --without_data_creation || { echo 'Failed to install pipeline_bench submodule'; exit 1; }
        echo "Warning - Pipeline-Bench data creation is not compatible with Python 3.10."
        echo "Installing tabrepo..."
        poetry run install-tabrepo || { echo 'Failed to install tabrepo submodule'; exit 1; }
        echo "Installing phem..."
        poetry run install-phem || { echo 'Failed to install phem submodule'; exit 1; }
    elif [[ "$python_version" == "3.9" ]]; then
        echo "Python 3.9 detected, setting up accordingly."
        # Install pipeline_bench submodule
        echo "Installing pipeline_bench..."
        poetry run install-pipeline_bench || { echo 'Failed to install pipeline_bench submodule'; exit 1; }
        echo "Warning - phem submodule is not compatible with Python 3.9."
    else
        echo "Unsupported Python version. Please use Python 3.9 or 3.10."
        exit 1
    fi

    poetry install
}


# Check if conda commands are available
check_conda

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
