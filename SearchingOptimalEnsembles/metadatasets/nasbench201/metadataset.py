from __future__ import annotations

import os
import pickle
from collections import defaultdict

import torch
from containers import Baselearner, get_arch_filename, get_arch_str

from ..base_metadataset import BaseMetaDataset


class NASBench201MetaDataset(BaseMetaDataset):
    def __init__(
        self,
        data_dir: str = "/home/zelaa/nes/data/nb201_preds",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 1,
        split: str = "valid",
        metric_name: str = "error",
    ):
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
        )

        self.feature_dim = 6
        with open("configs/arch_to_id.pkl", "rb") as f:
            self.arch_to_id_dict = pickle.load(f)

        self.load_all_data(nb201_data_path=data_dir)
        self._initialize()

    def load_baselearner(self, config: list[int]) -> Baselearner:
        # firstly retrieve the unique filename from the config
        arch_str = get_arch_str(config)
        arch_id = str(self.arch_to_id_dict[arch_str][0])
        filename = get_arch_filename(arch_id)

        # load the baselearner
        baselearner = Baselearner.load(self.data_dir, filename)
        return baselearner

    def load_all_data(self, nb201_data_path: str):
        self.benchmark: defaultdict = defaultdict(dict)

        filenames = [get_arch_filename(x[0]) for x in self.arch_to_id_dict.values()]
        for dataset in self.get_dataset_names():
            for seed in range(3):
                for filename in filenames:
                    directory = os.path.join(nb201_data_path, dataset, str(seed))
                    if f"{filename}.pt" in os.listdir(directory):
                        self.benchmark[dataset][seed][filename] = Baselearner.load(
                            directory, filename
                        )

    def get_hp_candidates_ids(self):
        return NotImplementedError

    def get_dataset_names(self) -> list[str]:
        return ["cifar10", "cifar100", "imagenet"]

    def set_state(self, dataset_name: str):
        raise NotImplementedError

    def evaluate_ensembles(
        self,
        ensembles: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
