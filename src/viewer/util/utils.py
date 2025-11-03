from __future__ import annotations
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# カラーサイクルのユーティリティ
# color = ColorCycle()() で20色が順番に取得される
# シングルトンなので、別のクラスから雑にアクセスをして問題ない
class ColorCycle:
    _instance: ColorCycle | None = None
    colors: npt.NDArray[np.floating]
    index: int

    def __new__(cls, n_colors: int = 20):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
            cls._instance.index = 0
        return cls._instance

    def __call__(self) -> npt.NDArray[np.floating]:
        color = self.colors[self.index % len(self.colors)]  
        self.index += 1
        return color

    def reset(self):
        self.index = 0
