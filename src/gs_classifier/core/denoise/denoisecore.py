"""点群のノイズ除去コアアルゴリズム。"""

import numpy as np
from sklearn.cluster import DBSCAN

from gs_classifier.core.kdtree import KDTree


class DenoiseCore:
    """2D 平面上で密度の低い点をノイズとして検出する。

    Attributes:
        points: 3D 点群 (N, 3)。
        kdtree: 円柱探索用の KDTree。
    """

    points: np.ndarray
    flatten_points: np.ndarray

    def __init__(self, points: np.ndarray) -> None:
        """DenoiseCore を初期化する。

        Args:
            points: 3D 点群 (N, 3)。
        """
        self.points = points
        self.kdtree = KDTree(points)

    def denoise_naive(self, radius: float = 0.1) -> np.ndarray:
        """全点を走査してノイズインデックスを返す。

        近傍点数が 10 未満の点をノイズとする。
        計算量が O(N) で非常に遅い。ばか時間かかるのでなし。

        Args:
            radius: 円柱探索の半径。

        Returns:
            ノイズ点のインデックス配列。
        """
        noise_indices = []
        for i in range(len(self.points)):
            num_points, indices, distances = self.kdtree.cylinder_search(
                self.points[i], radius
            )
            if num_points < 10:
                noise_indices.append(i)
            if i % 100000 == 0:
                print(f"denoise: {i} / {len(self.points)}")
        return np.array(noise_indices)

    def denoise_density(self, radius: float = 0.1) -> np.ndarray:
        """ランダムサンプリングで密度の低い領域を検出する。

        100万点をランダムにサンプリングし、周囲の点が
        100 未満の領域をノイズとして返す。

        Args:
            radius: 円柱探索の半径。

        Returns:
            ノイズ点のインデックス配列。
        """
        # NOTE: ふつうにランダムじゃなくて、メッシュみたいにサンプリングしたほうがいい
        noise_indices = []
        random_points = self.points[
            np.random.choice(len(self.points), size=1000000, replace=False)
        ]
        for i in range(len(random_points)):
            num_points, indices, distances = self.kdtree.cylinder_search(
                random_points[i], radius
            )
            if num_points < 100:
                noise_indices.extend(indices)

        return np.unique(noise_indices)

    def denoise_dbscan(self, radius: float = 0.1) -> np.ndarray:
        """DBSCAN で最大クラスタのインデックスを返す。

        Args:
            radius: DBSCAN の eps は radius * 2 で設定される。

        Returns:
            最大クラスタに属する点のインデックス配列。
        """
        dbscan = DBSCAN(eps=radius * 2, min_samples=50)
        labels = dbscan.fit_predict(self.points)
        unique, counts = np.unique(labels, return_counts=True)
        largest_label = unique[np.argmax(counts)]
        largest_indices = np.where(labels == largest_label)[0]
        return largest_indices
