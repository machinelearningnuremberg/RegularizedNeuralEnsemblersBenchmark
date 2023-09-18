from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import wandb
from typing_extensions import Literal

from ...metadatasets.base_metadataset import BaseMetaDataset
from ...samplers import SamplerMapping
from ...utils.common import instance_from_map
from ..base_searcher import BaseSearcher

class DivBO(BaseSearcher):
    """Diversity-aware Bayesian Optimization proposed in https://arxiv.org/pdf/2302.03255.pdf"""

    def __init__(self,
                 metadataset: BaseMetaDataset,
                 patience: int = 50,):

        super(DivBO, self).__init__(metadataset=metadataset, patience=patience)


    def suggest(self,
                max_num_pipelines: int = 1,):
        pass