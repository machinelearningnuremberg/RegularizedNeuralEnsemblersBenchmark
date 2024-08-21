#!/bin/bash

# Define the path where the pipeline_bench should be cloned
PIPELINE_BENCH_DIR="SearchingOptimalEnsembles/metadatasets/scikit_learn/pipeline_bench"
AUTO_SKLEARN_PATH="SearchingOptimalEnsembles/metadatasets/scikit_learn/pipeline_bench/pipeline_bench/auto-sklearn"

# Check command line for flags
if [[ "$1" == "--without_data_creation" ]]; then
    EXTRA_FLAG="without_data_creation"
else
    EXTRA_FLAG="soe_compatibility"
fi

# Check if the tabrepo directory already exists
if [ ! -d "$PIPELINE_BENCH_DIR" ]; then
    # Clone the repository from the tpot_bench branch and include submodules
    git clone --branch tpot_bench --recurse-submodules https://github.com/releaunifreiburg/Pipeline-Bench.git $PIPELINE_BENCH_DIR
else
    # If the directory exists, navigate to it and update submodules
    cd $PIPELINE_BENCH_DIR
    git submodule update --init --recursive
    cd -
fi

# Install Pipeline-Bench with the specified extras using Poetry
poetry add $PIPELINE_BENCH_DIR --extras $EXTRA_FLAG

echo "Pipeline-Bench has been installed with the $EXTRA_FLAG extras."
