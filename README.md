# Regularized Neural Ensemblers

<p align="center">
  <a href="https://github.com/machinelearningnuremberg/SearchingOptimalEnsembles">
    <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
    </a>&nbsp; <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/pytorch-2.0-orange?style=for-the-badge&logo=pytorch" alt="PyTorch Version" />
    </a>&nbsp;
    <a href="https://github.com/machinelearningnuremberg/SearchingOptimalEnsembles">
    <img src="https://img.shields.io/badge/open-source-9cf?style=for-the-badge&logo=Open-Source-Initiative" alt="Open Source" />
  </a>
</p>

## Introduction

Ensemble methods can significantly enhance the accuracy and robustness of machine learning models by combining multiple base learners. However, standard approaches like greedy or random ensembling often assume a constant weighting for each base model, which can limit expressiveness.

This repository explores **dynamic neural ensemblers**, where a neural network adaptively aggregates predictions from multiple candidate models. To address overfitting and low-diversity ensembles, we propose a simple but effective regularization strategy by randomly dropping base model predictions during training, ensuring a lower bound on ensemble diversity.

## Installation

We use [conda](https://www.anaconda.com/) for environment management and [Poetry](https://python-poetry.org/docs) for dependency installation.

### 1. Optional: Install miniconda and create an environment

```
# Download & install Miniconda (adjust for your operating system)
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p $HOME/.conda
rm install_miniconda.sh

# Initialize conda for your shell (e.g., bash or zsh)
~/.conda/bin/conda init
# Then restart your shell or source your rc file

# Create and activate the environment
conda create -n searching_optimal_ensembles python=3.10
conda activate searching_optimal_ensembles
```

### 2. Install poetry

````
curl -sSL https://install.python-poetry.org | python3 -
# Add Poetry to your PATH in ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"```

Consider appending `export PATH="$HOME/.local/bin:$PATH"` into `~/.zshrc` / `~/.bashrc`.
````

### 3. Install dependencies

```
bash setup.sh
```

This will install all dependencies into the Poetry environment.

Note: Due to the scikit-learn mismatch, the default installation support tabrepo and phem libraries; for running experiments with Pipeline-Bench metadataset using TPOT search space, install the dependencies with `bash setup.sh install-pipeline_bench`.

## Usage Example

Below are minimal examples for **random, greedy, and neural** ensemble methods. Each script showcases how to load the metadataset, sample (or build) ensembles, and evaluate performance. Adjust paths (e.g., DATA_DIR) as needed.

### Random Ensembling

```python
# SearchingOptimalEnsembles_experiments/random_ensemble_example.py
# Demonstrates how to build a random ensemble on a metadataset.

import torch

from SearchingOptimalEnsembles.posthoc.random_ensembler import RandomEnsembler
import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd

if __name__ == "__main__":
    data_version = "micro"
    metric_name = "nll"
    task_id = 0  # or any valid index
    DATA_DIR = "path/to/quicktune/predictions"

    metadataset = qmd.QuicktuneMetaDataset(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )
    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])

    # Initialize random sampler
    ensembler = RandomEnsembler(
        metadataset=metadataset,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Candidate pipelines (in practice, you'd sample or load these)
    X_obs = [[1], [2], [3], [4], [5], [6], [7], [8]]

    best_ensemble, best_metric = ensembler.sample(X_obs)
    print("Best random ensemble found:", best_ensemble)
    print("Random ensembler metric:", best_metric)
```

Run the script with `python SearchingOptimalEnsembles_experiments/random_ensemble_example.py`.

### Greedy Ensembling

```python
# SearchingOptimalEnsembles_experiments/greedy_ensemble_example.py
# Demonstrates how to build a greedy ensemble on a metadataset.

import torch

from SearchingOptimalEnsembles.posthoc.greedy_ensembler import GreedyEnsembler
import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd

if __name__ == "__main__":
    data_version = "micro"
    metric_name = "nll"
    task_id = 0  # or any valid index
    DATA_DIR = "path/to/quicktune/predictions"

    metadataset = qmd.QuicktuneMetaDataset(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )

    dataset_names = metadataset.get_dataset_names()
    metadataset.set_state(dataset_names[0])

    # Initialize greedy ensembler
    ensembler = GreedyEnsembler(
        metadataset=metadataset,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Candidate pipelines (in practice, you'd sample or load these)
    X_obs = [[1], [2], [3], [4], [5], [6], [7], [8]]

    best_ensemble, best_metric = ensembler.sample(X_obs)
    print("Greedy ensemble found:", best_ensemble)
    print("Greedy ensemble metric:", best_metric)
```

Run the script with `python SearchingOptimalEnsembles_experiments/greedy_ensemble_example.py`.

### Neural Ensembling

```python
# SearchingOptimalEnsembles_experiments/neural_ensemble_example.py
# Demonstrates how to train a neural ensemble on a metadataset.

import torch

from SearchingOptimalEnsembles.posthoc.neural_ensembler import NeuralEnsembler
import SearchingOptimalEnsembles.metadatasets.quicktune.metadataset as qmd

if __name__ == "__main__":
    data_version = "micro"
    metric_name = "nll"
    task_id = 0  # or any valid index
    DATA_DIR = "path/to/quicktune/predictions"

    metadataset = qmd.QuicktuneMetaDataset(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )

    dataset_names = metadataset.meta_splits["meta-test"]
    metadataset.set_state(dataset_names[task_id])

    # Initialize the neural ensembler
    neural_ensembler = NeuralEnsembler(
        metadataset=metadataset,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Candidate pipelines (in practice, you'd sample or load these)
    X_obs = [[1], [2], [3], [4], [5], [6], [7], [8]]

    # Now sample an ensemble with learned dynamic weights
    best_ensemble, best_metric = neural_ensembler.sample(X_obs)
    weights = neural_ensembler.get_weights(X_obs)

    _, metric_val, _, _ = metadataset.evaluate_ensembles_with_weights(
        ensembles=[best_ensemble], weights=weights
    )
    print("Best ensemble found by Neural Ensembler:", best_ensemble)
    print("Neural ensemble metric:", metric_val.item())
```

Run the script with `python SearchingOptimalEnsembles_experiments/neural_ensemble_example.py`.
