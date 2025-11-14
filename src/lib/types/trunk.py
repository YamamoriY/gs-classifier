from __future__ import annotations
from dataclasses import dataclass
from src.lib.types.gstype import GSplatData
import numpy as np

class TrunkLocation:
    trunk_locations: np.ndarray # (N, 3)
    def __init__(self, trunk_locations: np.ndarray):
        self.trunk_locations = trunk_locations

    # GSplatData から TrunkLocation を作成
    # gs.labels に幹のラベルが入っている想定、-1 は除外
    @classmethod
    def from_gs(cls, gs: GSplatData) -> TrunkLocation:
        centers = gs.centers
        labels = gs.labels
        unique_labels = np.unique(labels)
        trunk_locations = []
        for label in unique_labels:
            if label == -1:
                continue
            mask = labels == label
            trunk_locations.append(np.mean(centers[mask], axis=0))
        trunk_locations = np.array(trunk_locations)
        return cls(trunk_locations)