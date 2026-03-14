"""GSplatData に対するノイズ除去ラッパー。"""

import numpy as np

from gs_classifier.core.denoise.denoisecore import DenoiseCore
from gs_classifier.core.kdtree import KDTree
from gs_classifier.models import GSplatData


class NoiseRemover:
    """GSplatData からノイズを除去するラッパークラス。

    DenoiseCore を内部で利用し、DBSCAN や密度ベースの
    ノイズ除去を GSplatData 単位で実行する。

    Attributes:
        denoise_core: 低レベルのノイズ除去エンジン。
        gs: 処理対象の GSplatData。
    """

    denoise_core: DenoiseCore
    gs: GSplatData

    def __init__(self, gs: GSplatData) -> None:
        """NoiseRemover を初期化する。

        Args:
            gs: 処理対象の GSplatData。
        """
        self.denoise_core = DenoiseCore(gs.centers)
        self.gs = gs

    def run_2d_dbscan(
        self, radius: float = 0.1
    ) -> tuple[GSplatData, GSplatData]:
        """2D 平面上で DBSCAN を実行し、最大クラスタを抽出する。

        Args:
            radius: DBSCAN の eps 計算に使用する半径。

        Returns:
            (信号データ, ノイズデータ) のタプル。
        """
        print("start denoise 2d dbscan")
        indices = self.denoise_core.denoise_dbscan(radius)
        self.gs.labels = np.zeros_like(self.gs.labels)
        self.gs.labels[indices] = 1
        noise_gs = self.gs.split_by_label()[0]
        gs = self.gs.split_by_label()[1]
        return gs, noise_gs

    def run_3d_density(
        self, radius: float = 0.1, point_count: int = 200
    ) -> tuple[GSplatData, GSplatData]:
        """密度の低い点をノイズとして除去する。

        Args:
            radius: 円柱探索の半径。
            point_count: この点数未満の領域をノイズとする。

        Returns:
            (信号データ, ノイズデータ) のタプル。
        """
        # NOTE: ここ core と分離するべき
        print("start denoise 3d density")
        tmp = self.gs.centers.copy()
        tmp[:, 2] = tmp[:, 2] * 0.5
        kdtree = KDTree(tmp)
        for i in range(len(tmp)):
            num_points, indices, distances = kdtree.cylinder_search(
                tmp[i], radius
            )
            if num_points < point_count:
                self.gs.labels[i] = 1
        gs = self.gs.split_by_label()[0]
        noise_gs = self.gs.split_by_label()[1]
        return gs, noise_gs
