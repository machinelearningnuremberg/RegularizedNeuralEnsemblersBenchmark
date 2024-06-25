from __future__ import annotations

import pandas as pd

class BasePipelineSampler:
    def __init__(self):
        self.hps = None

    def process_hps(self, hps: pd.DataFrame):

        hps = pd.get_dummies(hps).astype(float)
        hps = hps.fillna(0).values

        return hps