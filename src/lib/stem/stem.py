from src.lib.types.types import GSplatData
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

    def dbscan_stem(self) -> GSplatData:
        # DBSCAN で区分
        dbscan = DBSCAN(eps=0.05, min_samples=50)
        tmp = self.gs.centers.copy()
        tmp[:, 2] = tmp[:, 2] * 0.1
        labels = dbscan.fit_predict(tmp)
        self.gs.labels = labels
        return self.gs

