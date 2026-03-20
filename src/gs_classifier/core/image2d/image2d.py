"""幹位置ベースの単木分割。"""

from __future__ import annotations

from gs_classifier.core.kdtree import KDTree
from gs_classifier.models import GSplatData, TrunkLocation


class Image2D:
    """幹位置を基準に点群を個々の木に分割する。

    Attributes:
        gs: 処理対象の GSplatData（コピー）。
        trunk_location: 幹の位置情報。
        kdtree: 探索用 KDTree。
    """

    def __init__(self, gs: GSplatData, trunk_location: TrunkLocation) -> None:
        """Image2D を初期化する。

        Args:
            gs: 処理対象の GSplatData。
            trunk_location: 幹の位置情報。
        """
        self.gs = gs.copy()
        self.trunk_location = trunk_location
        self.kdtree = KDTree(gs.centers)

    def segment_trees(self, radius: float = 1.0) -> list[GSplatData]:
        """幹の周辺 radius の範囲内にある点を各木に分類する。

        Args:
            radius: 幹からの最大距離。

        Returns:
            各木の GSplatData のリスト。
        """
        self.gs.reset_labels()
        for i in range(len(self.gs.centers)):
            num_points, indices, distances = self.kdtree.knn_search(
                self.gs.centers[i], 1
            )
            if num_points <= 0:
                print("Error ?")
                continue
            if distances[0] > radius:
                continue
            self.gs.labels[indices] = i + 1
        return self.gs.split_by_label()[1:]  # 0 は分類外なので除外
        # return self.gs  # ラベルで返したければこっち(0は除外すべし)

    def create_image(self, radius: float = 1.0) -> None:
        """2D 画像を生成する（未実装）。

        Args:
            radius: 幹からの最大距離。
        """
        self.segment_trees(radius)
        # 凍結中
