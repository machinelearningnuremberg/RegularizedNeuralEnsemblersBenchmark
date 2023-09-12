from __future__ import annotations

from typing import Callable

try:
    from .nasbench201.metadataset import NASBench201MetaDataset
except ImportError:
    NASBench201MetaDataset = None

from .quicktune.metadataset import QuicktuneMetaDataset
from .scikit_learn.metadataset import ScikitLearnMetaDataset

MetaDatasetMapping: dict[str, Callable] = {
    "scikit-learn": ScikitLearnMetaDataset,
    "nasbench201": NASBench201MetaDataset,
    "quicktune": QuicktuneMetaDataset,
}
