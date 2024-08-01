from __future__ import annotations

from typing import Callable

try:
    from .tabrepo.metadataset import TabRepoMetaDataset
except Exception as e:
    print(f"Error importing TabRepoMetaDataset: {e}")
    TabRepoMetaDataset = None

from .quicktune.metadataset import QuicktuneMetaDataset
from .ftc.metadataset import FTCMetaDataset

try:
    from .scikit_learn.metadataset import ScikitLearnMetaDataset
except Exception as e:
    print(f"Error importing ScikitLearnMetaDataset: {e}")
    ScikitLearnMetaDataset = None
from .nasbench201.metadataset import NASBench201MetaDataset

try:
    from .custom.openml_metadataset import OpenMLMetaDataset
except Exception as e:
    print(f"Error importing Openml Dataset: {e}")
    OpenMLMetaDataset = None

MetaDatasetMapping: dict[str, Callable] = {
    "scikit-learn": ScikitLearnMetaDataset,
    "nasbench201": NASBench201MetaDataset,
    "quicktune": QuicktuneMetaDataset,
    "tabrepo": TabRepoMetaDataset,
    "ftc": FTCMetaDataset,
    "openml": OpenMLMetaDataset
}
