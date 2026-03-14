"""HSV 色空間ベースの葉分類コアアルゴリズム。"""

import numpy as np

from gs_classifier.models import GSplatData


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB 配列を HSV に変換する。

    入力にシグモイドを適用してから変換する。

    Args:
        rgb: RGB 値の配列 (N, 3)。

    Returns:
        HSV 値の配列 (N, 3)、各値は 0-1 に正規化。
    """
    # シグモイドを作用（.ply のデフォルトはシグモイド前の状態？）
    rgb = 1 / (1 + np.exp(-rgb))

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    v = maxc

    deltac = maxc - minc
    s = np.where(maxc != 0, deltac / maxc, 0)

    # Hue の計算
    h = np.zeros_like(v)

    idx = maxc == r
    h[idx] = ((g[idx] - b[idx]) / deltac[idx]) % 6

    idx = maxc == g
    h[idx] = ((b[idx] - r[idx]) / deltac[idx]) + 2

    idx = maxc == b
    h[idx] = ((r[idx] - g[idx]) / deltac[idx]) + 4

    h[deltac == 0] = 0
    h = h / 6.0  # 0-1 に正規化

    return np.stack([h, s, v], axis=1)


class LeafClassifier:
    """HSV 色空間の閾値で葉を分類する。

    緑・青系と黄色系の色相をマスクして葉を検出する。
    単木抽出の前処理として使用する。

    Attributes:
        gs: 処理対象の GSplatData。
    """

    gs: GSplatData

    def __init__(self, gs: GSplatData) -> None:
        """LeafClassifier を初期化する。

        Args:
            gs: 処理対象の GSplatData。
        """
        self.gs = gs

    def classify_leaf(self) -> np.ndarray:
        """葉のインデックスを返す。

        Returns:
            葉と判定された点のインデックス配列。
        """
        hsv = rgb_to_hsv(self.gs.rgbs)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        # 葉を消す
        green_mask = (
            (h >= 0.17)
            & (h <= 0.67)  # 緑・青系の色相
            & (s >= 0.02)  # 彩度
            & (v >= 0.05)  # 明度（暗めの色も含める）
        )

        # 日が当たって黄色っぽい葉を消す
        yellow_mask = (
            (h >= 0.08)
            & (h <= 0.17)  # 黄色系の色相
            & (s >= 0.1)  # 彩度
            & (v >= 0.2)  # ある程度の明度
        )

        # マスクを結合
        leaf_mask = green_mask | yellow_mask
        leaf_indices = np.where(leaf_mask)[0]
        return leaf_indices
