from __future__ import annotations

from ..base_metadataset import BaseMetaDataset


class QuicktuneMetaDataset(BaseMetaDataset):
    def __init__(
        self,
        data_dir: str,
        data_pct: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ):
        raise NotImplementedError
