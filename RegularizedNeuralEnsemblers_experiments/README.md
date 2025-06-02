# Experiments & Reproducibility

This directory contains **SLURM-ready pipelines**, configuration grids, and reporting utilities that reproduce the paper’s results.

---

## Folder Map
```text
RegularizedNeuralEnsemblers_experiments/
├── cluster_scripts/
│   └── slurm_pineda/
│       ├── bash_args/                # auto-generated job arguments
│       ├── experiments_confs/        # YAML grids (RQ1, …)
│       ├── generate_commands.py   # [0] create bash_args/*
│       ├── all_run_RQ1.sh         # [1] launch full sweep
│       ├── per_project_run_RQ1.sh # [2] per-dataset launcher
│       └── run.sh                 # [3] SLURM job wrapper
├── reporter_configs/
│   ├── report1.yml
│   └── report2.yml
├── report.py                       # [5] aggregate logs → tables
├── main.py                        # [4] entry point for training
└── examples/                       # minimal random/greedy/neural scripts
```

---

## Examples

*
*
*
---

## Workflow

### 1 · Generate Job Commands
```bash
python cluster_scripts/slurm_pineda/generate_commands.py \
       --exp_dir cluster_scripts/slurm_pineda/experiments_confs
```

### 2 · Submit a Full Sweep
```bash
bash cluster_scripts/slurm_pineda/all_run_RQ1.sh
```
This calls **[2]** (`per_project_run_RQ1.sh`) which loops over dataset IDs and dispatches **[3]** (`run.sh`) to SLURM.

### 3 · Collect Results
```bash
python report.py --config reporter_configs/report1.yml
```
Produces LaTeX-ready tables and JSON summaries.

---

## Single Local Run Example
```bash
python main.py \
  --ensembler_name neural \
  --no_wandb \
  --dataset_id 0 \
  --project_name test \
  --num_iterations 100 \
  --metric_name error \
  --metadataset_name ftc \
  --data_version extended \
  --ne_epochs 1000 \
  --ne_batch_size 256 \
  --ne_hidden_dim 32 \
  --ne_net_mode stacking
```

---

## Tips
* Add `--searcher_name None` to disable hyper-parameter search when debugging.
* Edit `cluster_scripts/.../run.sh` to match your GPU/CPU partition before submitting.
