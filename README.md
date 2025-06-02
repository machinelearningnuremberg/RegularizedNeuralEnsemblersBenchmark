# Regularized Neural Ensemblers

<p align="center">
  <a href="https://github.com/machinelearningnuremberg/RegularizedNeuralEnsemblers">
    <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
  </a>&nbsp;
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/pytorch-2.0-orange?style=for-the-badge&logo=pytorch" alt="PyTorch Version" />
  </a>&nbsp;
  <a href="https://github.com/machinelearningnuremberg/RegularizedNeuralEnsemblers">
    <img src="https://img.shields.io/badge/open--source-9cf?style=for-the-badge&logo=open-source-initiative" alt="Open Source" />
  </a>
</p>

## Abstract
Ensemble methods are renowned for boosting the accuracy and robustness of machine-learning models by aggregating complementary base learners. Yet classic approaches—such as greedy or random ensembling—assume *fixed* weights for every sample, limiting expressiveness. We revisit **dynamic ensembling** with a lightweight MLP that learns *instance-wise* weights over candidate models. To curb low-diversity ensembles and overfitting, we introduce **Random Base-Model Dropout**, randomly masking model predictions during training and enforcing a diversity lower-bound. Experiments across vision, NLP, and tabular tasks show that **Regularized Neural Ensemblers (RNE)** match or surpass strong post-hoc and stacking baselines while remaining data-efficient and easy to integrate.

---

## At a Glance

| &nbsp; | What | Where |
|-------|-------|-------|
| **Datasets** | loaders & download instructions | [`metadatasets/`](./RegularizedNeuralEnsemblers/metadatasets/README.md) |
| **Post-hoc algorithms** | random, greedy, CMA-ES, Akaike, DES, *etc.* | [`posthoc/`](./RegularizedNeuralEnsemblers/posthoc/README.md) |
| **Experiments** | exampler, SLURM scripts, configs, reports | [`experiments/`](./RegularizedNeuralEnsemblers_experiments/README.md) |

---

## Quick Start

### 1 · Install with Conda + Poetry
```bash
conda create -n rne python=3.10 -y
conda activate rne

curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

bash setup.sh          # or: bash setup.sh install-pipeline_bench
```

### 2 · Train a Neural Ensembler in <10 Lines
```python
from pathlib import Path
import torch
from RegularizedNeuralEnsemblers.posthoc.neural_ensembler import NeuralEnsembler
from RegularizedNeuralEnsemblers.metadatasets.quicktune.metadataset import QuicktuneMetaDataset

DATA_DIR = Path("/path/to/quicktune/predictions")
meta = QuicktuneMetaDataset(data_dir=DATA_DIR, metric_name="nll", data_version="micro")
meta.set_state(meta.meta_splits["meta-test"][0])          # pick a dataset

ensembler = NeuralEnsembler(meta, device="cuda" if torch.cuda.is_available() else "cpu")
ensemble, _ = ensembler.sample([[1], [2], [3]])           # candidate IDs → ensemble
```
Full runnable scripts (random, greedy, neural) live in [`examples/`](./RegularizedNeuralEnsemblers_experiments/).

<!-- --- -->
<!--
## Installation (Detailed)

1. **Conda** (optional) – create & activate an isolated environment.
2. **Poetry** – installs pinned dependencies (PyTorch 2.0+, scikit-learn, …).
3. **Datasets** – some corpora (e.g. *scikit-learn*, *QuickTune*) must be downloaded manually; see [`metadatasets/README.md`](./RegularizedNeuralEnsemblers/metadatasets/README.md). -->

---

## Repository Layout
```text
RegularizedNeuralEnsemblersBenchmark/
├── RegularizedNeuralEnsemblers/          # main Python package
│   ├── __init__.py
│   ├── metadatasets/                     # dataset loaders & metadata
│   ├── posthoc/                          # ensembling algorithms
│   ├── samplers/                         # base-model sampling strategies
│   └── searchers/                        # hyper-parameter / architecture search
│   └── ...                               # other modules (e.g. plot, tests, etc.)
│
├── RegularizedNeuralEnsemblers_experiments/  # reproducible pipelines & SLURM helpers
│   ├── main.py                           # entry point for training
│   ├── report.py                         # result aggregation
│   ├── examples/                         # runnable scripts
│   └── cluster_scripts/                  # SLURM submission helpers
│
├── setup.sh                              # dependency installer
└── README.md                             # ← you are here

```

---

## Citing
```bibtex

```

---

## License
Apache 2.0 – see [LICENSE](./LICENSE) for full text.
