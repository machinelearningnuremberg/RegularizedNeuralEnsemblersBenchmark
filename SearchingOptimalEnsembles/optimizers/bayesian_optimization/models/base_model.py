from __future__ import annotations

from abc import abstractmethod


class BaseModel:
    def __init__(self, metadataset):
        super().__init__()

        self.metadataset = metadataset

        self.observed_ids = {}
        self.pending_ids = {}
        for dataset_name in self.metadataset.dataset_names:
            self.observed_ids[dataset_name] = []
            self.pending_ids[dataset_name] = metadataset.get_hp_candidates(dataset_name)

    def _observe(self, dataset_name: str, new_id: int):
        self.observed_ids[dataset_name].append(new_id)
        self.pending_ids[dataset_name].remove(new_id)

    @abstractmethod
    def load_checkpoint(self):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self):
        raise NotImplementedError

    @abstractmethod
    def observe(self, dataset_name: str, new_id: int):
        self._observe(dataset_name, new_id)
        self._fit()

        raise NotImplementedError

    @abstractmethod
    def _fit(self):
        raise NotImplementedError

    @abstractmethod
    def generate_candidates(self):
        raise NotImplementedError

    @abstractmethod
    def get_next(self):
        raise NotImplementedError
