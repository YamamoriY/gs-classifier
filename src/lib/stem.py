from src.lib.types import GSplatData
import numpy as np
from sklearn.cluster import DBSCAN
from src.lib.kdtree import KDTree

class StemDetector:
    def __init__(self, gs: GSplatData):
        self.gs = gs

    def detect_stem_old(self) -> np.ndarray:
        # DBSCAN で stem を分けていく
        # eps: 近傍距離を大きくして、より大雑把なクラスタリング
        # min_samples: 最小点数を大きくして、小さなクラスタを排除
        dbscan = DBSCAN(eps=0.1, min_samples=50)
        labels = dbscan.fit_predict(self.gs.centers)
        return labels

    def dbscan_stem(self) -> np.ndarray:
        # 密度の低い点を消す
        tmp = self.gs.centers.copy()
        tmp[:, 2] = tmp[:, 2] * 0.5   # z方向に縮小
        kdtree = KDTree(tmp)
        for i in range(len(tmp)):
            num_points, indices, distances = kdtree.cylinder_search(tmp[i], 0.1)
            if num_points < 200:
                self.gs.labels[i] = 1
        self.gs_object = self.gs.split_by_label()[0]
        self.gs_noise = self.gs.split_by_label()[1]

        # DBSCAN で区分
        dbscan = DBSCAN(eps=0.05, min_samples=50)
        tmp = self.gs_object.centers.copy()
        tmp[:, 2] = tmp[:, 2] * 0.1
        labels = dbscan.fit_predict(tmp)
        self.gs_object.labels = labels
        return self.gs_object, self.gs_noise

