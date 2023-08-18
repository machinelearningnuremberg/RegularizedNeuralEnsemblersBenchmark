from __future__ import annotations

from abc import abstractmethod


class BaseModel:
    def __init__(self, metadataset):
        super().__init__()

        self.metadataset = metadataset


    def initialize_model(self, dataset_name: str):
        self.observed_ids = []
        self.pending_ids = self.metadataset.get_hp_candidates(dataset_name)

    def _observe(self, new_id: int):
        self.observed_ids.append(new_id)
        self.pending_ids.remove(new_id)

    @abstractmethod
    def meta_train(self):
        raise NotImplementedError

    @abstractmethod
    def meta_val(self):
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self):
        raise NotImplementedError

    @abstractmethod
    def observe(self, new_id: int):
        self._observe(new_id)
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
