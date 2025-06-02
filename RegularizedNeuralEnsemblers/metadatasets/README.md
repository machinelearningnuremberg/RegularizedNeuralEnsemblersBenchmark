# Metadatasets

This folder provides lightweight **loader classes** plus pointers to large prediction corpora that must be downloaded separately.

| Dataset | Domain | Default Versions† | Download Link |
|---------|--------|------------------|---------------|
| **scikit-learn** | tabular | `micro` | <https://placeholder-scikit-url.com> |
| **QuickTune** | tabular | `micro`, `mini`, `extended` | <https://placeholder-quicktune-url.com> |
| **NASBench-201** | vision | `micro` | <https://placeholder-nb201-url.com> |
| **TabRepo** | tabular | *tba* | *(auto-downloaded)*  |
| **FTC** | nlp | `extended` | *(auto-downloaded)* |
| **OpenML-20** | tabular | single version | *(auto-downloaded)* |

† If no version is specified, loaders default to **`micro`** for quick prototyping.


## Expected Directory Layout
```text
/data/
└──  scikit-learn/
│   ├── micro/
│   ├── mini/
│   └── extended/
├── quicktune/
├── nasbench201/
```

Point your code to the root:
```python
from pathlib import Path
DATA_DIR = Path("/data/scikit-learn")  # adjust as needed
```
