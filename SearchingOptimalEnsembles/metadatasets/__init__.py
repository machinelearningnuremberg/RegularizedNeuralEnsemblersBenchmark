from __future__ import annotations

from typing import Callable

try:
    from .tabrepo.metadataset import TabRepoMetaDataset
except ImportError:
    TabRepoMetaDataset = None

from .quicktune.metadataset import QuicktuneMetaDataset
from .scikit_learn.metadataset import ScikitLearnMetaDataset
from .nasbench201.metadataset import NASBench201MetaDataset

MetaDatasetMapping: dict[str, Callable] = {
    "scikit-learn": ScikitLearnMetaDataset,
    "nasbench201": NASBench201MetaDataset,
    "quicktune": QuicktuneMetaDataset,
    "tabrepo": TabRepoMetaDataset,  

}
