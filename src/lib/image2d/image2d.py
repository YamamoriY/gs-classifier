from __future__ import annotations
from src.lib.types.gstype import GSplatData
from src.lib.types.trunk import TrunkLocation
from src.lib.kdtree import KDTree
import numpy as np
import viser

class Image2D:
    def __init__(self, gs: GSplatData, trunk_location: TrunkLocation):
        self.gs = gs.copy()
        self.trunk_location = trunk_location
        self.kdtree = KDTree(gs.centers)

    # 幹の周辺 radius の範囲内にある点を分類
    def segment_trees(self, radius: float = 1.0) -> list[GSplatData]:
        self.gs.reset_labels()
        for i in range(len(self.gs.centers)):
            num_points, indices, distances = self.kdtree.knn_search(self.gs.centers[i], radius)
            if num_points <= 0:
                print("Error ?")
                continue
            if distances[0] > radius:
                continue
            self.gs.labels[indices] = i + 1
        return self.gs.split_by_label()[1:] # 0 は分類外なので除外
        # return self.gs  # ラベルで返したければこっち(0は除外すべし)

    # 別の方針でレンダリングしたほうが楽そうなのでやめる
    def create_image(self, radius: float = 1.0):
        trees_gs = self.segment_trees(radius)
        # 凍結中




