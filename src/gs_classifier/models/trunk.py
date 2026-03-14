"""幹位置データクラス。"""

from __future__ import annotations

import json

import numpy as np

from gs_classifier.models.gsplat import GSplatData


class TrunkLocation:
    """幹の位置情報を保持するデータクラス。

    Attributes:
        trunk_locations: 幹の中心座標 (N, 3)。
    """

    trunk_locations: np.ndarray  # (N, 3)

    def __init__(self, trunk_locations: np.ndarray) -> None:
        """TrunkLocation を初期化する。

        Args:
            trunk_locations: 幹の中心座標 (N, 3)。
        """
        self.trunk_locations = trunk_locations

    def coordinate_transform(self, R: np.ndarray) -> TrunkLocation:
        """座標変換を適用する。

        Args:
            R: 3x3 の回転行列。

        Returns:
            自身（メソッドチェーン用）。
        """
        self.trunk_locations = self.trunk_locations @ R
        return self

    @classmethod
    def from_gs(cls, gs: GSplatData) -> TrunkLocation:
        """GSplatData の各ラベルの重心から TrunkLocation を作成する。

        ラベル -1 は除外される。

        Args:
            gs: 幹のラベルが付与された GSplatData。

        Returns:
            幹位置の TrunkLocation。
        """
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

    def save_to_json(self, path: str) -> None:
        """JSON ファイルに保存する。

        Args:
            path: 出力先のファイルパス。
        """
        data = []
        for i, trunk_location in enumerate(self.trunk_locations):
            data.append(
                {
                    "name": f"trunk_{i}",
                    "x": float(trunk_location[0]),
                    "y": float(trunk_location[1]),
                    "z": float(trunk_location[2]),
                }
            )
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
