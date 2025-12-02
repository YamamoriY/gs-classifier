from src.lib.types.gstype import GSplatData
import numpy as np
from sklearn.cluster import DBSCAN
from src.lib.kdtree import KDTree
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
import scipy.stats

# 幹抽出のクラス
# 地上高 1~2m 程度の切り出された範囲を受け取ることを想定
class TrunkClassifier:
    def __init__(self, gs: GSplatData):
        self.gs = gs

    def run(self) -> GSplatData:
        self.dbscan_trunk()
        self.is_trunk_check_range()
        self.is_trunk_dbscan()
        # self.is_trunk_fit_line()  # このへん微妙だったので不採用中
        # self.is_trunk_std()
        return self.gs

    def detect_trunk_old(self) -> np.ndarray:
        # DBSCAN で trunk を分けていく
        # eps: 近傍距離を大きくして、より大雑把なクラスタリング
        # min_samples: 最小点数を大きくして、小さなクラスタを排除
        dbscan = DBSCAN(eps=0.1, min_samples=50)
        labels = dbscan.fit_predict(self.gs.centers)
        return labels

    # 基本。DBSCANで区分。ノイズは -1
    def dbscan_trunk(self) -> GSplatData:
        dbscan = DBSCAN(eps=0.05, min_samples=50)
        tmp = self.gs.centers.copy()
        tmp[:, 2] = tmp[:, 2] * 0.1     # 高さ方向の距離の重みを小さくする（平面に圧縮した状態で処理したい）
        labels = dbscan.fit_predict(tmp)
        self.gs.labels = labels
        return self.gs

    # z軸のみ取り出してdbscan、クラスタ数が1つでなければノイズとする
    def is_trunk_dbscan(self) -> GSplatData:
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
    def is_trunk_fit_line(self, slope_threshold_degree: float = 20) -> GSplatData:
        line = pyrsc.Line()
        unique_labels = np.unique(self.gs.labels)
        for label in unique_labels:
            if label == -1:
                continue
            mask = self.gs.labels == label
            centers = self.gs.centers[mask]
            slope, axis, inliers = line.fit(centers, thresh=0.15, maxIteration=200)
            inlier_ratio = len(inliers) / len(centers)
            theta = np.arccos(np.abs(slope[2])) * 180 / np.pi
            if theta > slope_threshold_degree or inlier_ratio < 0.4:
                print(f"killed label {label}")
                print(f"inlier ratio: {inlier_ratio}")
                print(f"theta: {theta}")
                self.gs.labels[mask] = -1
        return self.gs

    # 標準偏差が変なヤツをノイズとする
    def is_trunk_std(self) -> GSplatData:
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
            expected = np.full_like(counts, len(points_z)/20, dtype=float)
            stat, p = scipy.stats.chisquare(counts, f_exp=expected)
            
        return self.gs