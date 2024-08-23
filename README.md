# Searching Optimal Ensembles

<p align="center">
  <a href="https://github.com/releaunifreiburg/MetaLearningEnsembles">
    <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
  </a>&nbsp;
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/pytorch-2.0-orange?style=for-the-badge&logo=pytorch" alt="PyTorch Version" />
  </a>&nbsp;
  <a href="https://github.com/releaunifreiburg/MetaLearningEnsembles">
    <img src="https://img.shields.io/badge/open-source-9cf?style=for-the-badge&logo=Open-Source-Initiative" alt="Open Source" />
  </a>
  <!-- <a href="https://github.com/releaunifreiburg/MetaLearningEnsembles">
    <img src="https://img.shields.io/github/stars/releaunifreiburg/MetaLearningEnsembles=for-the-badge&logo=github" alt="GitHub Repo Stars" />
  </a> -->
</p>

## Contributing

### 1. Optional: Install miniconda and create an environment

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p $HOME/.conda  # Change to place of preference
rm install_miniconda.sh
```

Consider running `~/.conda/bin/conda init` or `~/.conda/bin/conda init zsh`.

Create the environment and activate it

```
conda create -n searching_optimal_ensembles python=3.10
conda activate searching_optimal_ensembles
```

### 2. Install poetry

First, [install poetry](https://python-poetry.org/docs), e.g., via

```
curl -sSL https://install.python-poetry.org | python3 -
```

Consider appending `export PATH="$HOME/.local/bin:$PATH"` into `~/.zshrc` / `~/.bashrc`.

### 3. Install dependencies

```
bash setup.sh
```

This will install all dependencies into the poetry environment. Due to the scikit-learn mismatch, the default installation support tabrepo and phem libraries; for running experiments with Pipeline-Bench metadataset using TPOT search space, install the dependencies with `bash setup.sh install-pipeline_bench`.

To install a new dependency use `poetry add dependency` and commit the updated `pyproject.toml` to git.

### 4. Activate pre-commit

```
pre-commit install
```

Consider appending `--no-verify` to your urgent commits to disable checks.
