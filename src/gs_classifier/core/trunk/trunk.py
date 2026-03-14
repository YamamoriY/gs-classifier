"""DBSCAN ベースの幹分類アルゴリズム。"""

import numpy as np
import pyransac3d as pyrsc
import scipy.stats
from sklearn.cluster import DBSCAN

from gs_classifier.models import GSplatData


class TrunkClassifier:
    """地上高 1-2m 程度の切り出された点群から幹を分類する。

    DBSCAN で点群をクラスタリングし、高さ方向の範囲や
    クラスタ数で幹でないクラスタを除去する。

    Attributes:
        gs: 処理対象の GSplatData。
    """

    def __init__(self, gs: GSplatData) -> None:
        """TrunkClassifier を初期化する。

        Args:
            gs: 処理対象の GSplatData。
        """
        self.gs = gs

    def run(self) -> GSplatData:
        """幹分類パイプラインを実行する。

        Returns:
            ラベル付けされた GSplatData（-1 はノイズ）。
        """
        self.dbscan_trunk()
        self.is_trunk_check_range()
        self.is_trunk_dbscan()
        # self.is_trunk_fit_line()  # このへん微妙だったので不採用中
        # self.is_trunk_std()
        return self.gs

    def dbscan_trunk(self) -> GSplatData:
        """DBSCAN で点群をクラスタリングする。

        高さ方向の距離の重みを 0.1 倍にして、XY 平面上での
        クラスタリングに近い結果を得る。

        Returns:
            ラベル付けされた GSplatData。
        """
        dbscan = DBSCAN(eps=0.05, min_samples=50)
        tmp = self.gs.centers.copy()
        tmp[:, 2] = (
            tmp[:, 2] * 0.1
        )  # 高さ方向の距離の重みを小さくする（平面に圧縮した状態で処理したい）
        labels = dbscan.fit_predict(tmp)
        self.gs.labels = labels
        return self.gs

    def is_trunk_dbscan(self) -> GSplatData:
        """Z 軸のみで DBSCAN し、クラスタ数が 1 でなければノイズにする。

        Returns:
            フィルタ後の GSplatData。
        """
        dbscan = DBSCAN(eps=0.15, min_samples=50)
        unique_labels = np.unique(self.gs.labels)
        for label in unique_labels:
            if label == -1:
                continue
            tmp = self.gs.centers[self.gs.labels == label]
            tmp = tmp[:, 2].reshape(-1, 1)  # z座標のみを2次元配列に変換
            labels = dbscan.fit_predict(tmp)
            if len(np.unique(labels)) > 1:
                self.gs.labels[self.gs.labels == label] = -1
            print(f"label {label} cluster: {len(np.unique(labels))}")
        return self.gs

    # 使えるが、fit_std でいい感じがするので不採用
    # あまりに直線形から外れたラベルをノイズとして -1 にする
    # 円柱フィットよりも直線フィットの方が精度が高いので、こちらを使用
    def is_trunk_fit_line(
        self, slope_threshold_degree: float = 20
    ) -> GSplatData:
        """直線フィットで幹らしくないクラスタをノイズにする。

        Args:
            slope_threshold_degree: 鉛直からの角度閾値（度）。

        Returns:
            フィルタ後の GSplatData。
        """
        line = pyrsc.Line()
        unique_labels = np.unique(self.gs.labels)
        for label in unique_labels:
            if label == -1:
                continue
            mask = self.gs.labels == label
            centers = self.gs.centers[mask]
            slope, axis, inliers = line.fit(
                centers, thresh=0.15, maxIteration=200
            )
            inlier_ratio = len(inliers) / len(centers)
            theta = np.arccos(np.abs(slope[2])) * 180 / np.pi
            if theta > slope_threshold_degree or inlier_ratio < 0.4:
                print(f"killed label {label}")
                print(f"inlier ratio: {inlier_ratio}")
                print(f"theta: {theta}")
                self.gs.labels[mask] = -1
        return self.gs

    def is_trunk_std(self) -> GSplatData:
        """標準偏差が異常なクラスタをノイズにする。

        Returns:
            フィルタ後の GSplatData。
        """
        unique_labels = np.unique(self.gs.labels)
        stds = []  # (N, 3)
        for label in unique_labels:
            if label == -1:
                continue
            mask = self.gs.labels == label
            stds.append(np.std(self.gs.centers[mask], axis=0))
        stds = np.array(stds)
        mean_std = np.mean(stds, axis=0)
        std_std = np.std(stds, axis=0)
        for i in range(len(stds)):
            for j in range(3):
                if np.abs(stds[i][j] - mean_std[j]) > std_std[j]:
                    self.gs.labels[unique_labels[i] == self.gs.labels] = -1
                    break
        return self.gs

    # これはよさげ
    # 高さ方向の範囲が狭いものをノイズに
    def is_trunk_check_range(self) -> GSplatData:
        """高さ方向の範囲が狭いクラスタをノイズにする。

        高さの範囲が 0.8 未満のクラスタを除去する。

        Returns:
            フィルタ後の GSplatData。
        """
        unique_labels = np.unique(self.gs.labels)
        for label in unique_labels:
            if label == -1:
                continue
            mask = self.gs.labels == label
            points = self.gs.centers[mask]
            points_z = points[:, 2]
            points_range = np.max(points_z) - np.min(points_z)
            print(f"label: {label}, points_range: {points_range}")
            if points_range < 0.8:
                self.gs.labels[mask] = -1
            counts, edges = np.histogram(points_z, bins=20)
            expected = np.full_like(counts, len(points_z) / 20, dtype=float)
            stat, p = scipy.stats.chisquare(counts, f_exp=expected)

        return self.gs
