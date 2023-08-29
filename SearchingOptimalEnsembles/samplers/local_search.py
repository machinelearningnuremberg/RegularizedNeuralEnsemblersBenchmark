from __future__ import annotations

import numpy as np
import torch

from ..metadatasets.base_metadataset import BaseMetaDataset
from ..utils.common import move_to_device
from .base_sampler import BaseSampler
