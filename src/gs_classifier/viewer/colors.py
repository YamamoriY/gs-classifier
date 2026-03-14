"""カラーサイクルユーティリティ。"""
# カラーサイクルのユーティリティ
# color = ColorCycle()() で20色が順番に取得される
# シングルトンなので、別のクラスから雑にアクセスをして問題ない

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class ColorCycle:
    """20 色のカラーサイクルを提供するシングルトン。

    呼び出すたびに次の色を返す。複数クラスから
    同じインスタンスにアクセスしても色が重複しない。

    Attributes:
        colors: カラーパレット (N, 4)。
        index: 現在のインデックス。
    """

    _instance: ColorCycle | None = None
    colors: npt.NDArray[np.floating]
    index: int

    def __new__(cls, n_colors: int = 20) -> ColorCycle:
        """シングルトンインスタンスを返す。

        Args:
            n_colors: パレットの色数（初回のみ有効）。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
            cls._instance.index = 0
        return cls._instance

    def __call__(self) -> npt.NDArray[np.floating]:
        """次の色を返す。

        Returns:
            RGBA 色の配列 (4,)。
        """
        color = self.colors[self.index % len(self.colors)]
        self.index += 1
        return color

    def reset(self) -> None:
        """インデックスを 0 にリセットする。"""
        self.index = 0
