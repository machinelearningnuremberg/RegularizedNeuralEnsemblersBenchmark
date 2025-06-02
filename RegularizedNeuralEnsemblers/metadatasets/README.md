# Metadatasets

This folder provides lightweight **loader classes** plus pointers to large prediction corpora that must be downloaded separately.

| Dataset | Domain | Default Versions† | Download Link |
|---------|--------|------------------|---------------|
| **scikit-learn** | tabular | `micro` | <https://placeholder-scikit-url.com> |
| **QuickTune** | tabular | XXXXXXX | <https://placeholder-quicktune-url.com> |
| **NASBench-201** | vision | `micro` | <https://placeholder-nb201-url.com> |
| **TabRepo** | tabular | XXXXXX | *(auto-downloaded)*  |
| **FTC** | nlp | XXXXXX | *(auto-downloaded)* |
| **OpenML-20** | tabular | XXXXXX | *(auto-downloaded)* |

† If no version is specified, loaders default to **`micro`** for quick prototyping.


## Expected Directory Layout
```text
/data/
└── quicktune/
├── scikit-learn/
├── nasbench201/
```
