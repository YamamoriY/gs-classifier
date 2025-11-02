from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt
import viser
from viser import GaussianSplatHandle

import matplotlib.pyplot as plt

# 未テスト
# カラーサイクル
# 20色が順に呼び出される
# get_color = ColorCycle()
# color = get_color() 
# シングルトンなので、別のクラスから同様のアクセスをしても問題ない
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

@dataclass
class GsplatData:
    name: str
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]

    def print_shape(self):
        print(f"centers: {self.centers.shape}")
        print(f"rgbs: {self.rgbs.shape}")
        print(f"opacities: {self.opacities.shape}")
        print(f"covariances: {self.covariances.shape}")
        print(f"x range: {np.min(self.centers[:, 0])} to {np.max(self.centers[:, 0])}")
        print(f"y range: {np.min(self.centers[:, 1])} to {np.max(self.centers[:, 1])}")
        print(f"z range: {np.min(self.centers[:, 2])} to {np.max(self.centers[:, 2])}")

class GSplatMode(Enum):
    NORMAL = "normal"
    POINTS_VIEW = "points_view"     

class GSplatHandle:
    gsplat: GaussianSplatHandle
    gsplat_points_view: GaussianSplatHandle
    current_mode: GSplatMode
    visible_checkbox: viser.GuiCheckboxHandle | None
    def __init__(self, data: GsplatData, server: viser.ViserServer):
        self.current_mode = GSplatMode.NORMAL
        # 通常
        self.gsplat = server.scene.add_gaussian_splats(
            name=f"{data.name}",
            centers=data.centers,
            rgbs=data.rgbs,
            opacities=data.opacities,
            covariances=data.covariances,
        )
        # ポイントビュー用
        cov_points_view = data.covariances.copy()
        cov_points_view[:, :3, :3] = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
        self.gsplat_points_view = server.scene.add_gaussian_splats(
            name=f"{data.name}_points_view",
            centers=data.centers,
            rgbs=data.rgbs,
            opacities=data.opacities,
            covariances=cov_points_view,
        )
        self.gsplat_points_view.visible = False
        self.visible_checkbox = None

    def change_mode(self, mode: GSplatMode):
        self.current_mode = mode
        if mode == GSplatMode.NORMAL:
            self.gsplat.visible = True
            self.gsplat_points_view.visible = False
        elif mode == GSplatMode.POINTS_VIEW:
            self.gsplat.visible = False
            self.gsplat_points_view.visible = True
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # 内部用
    def _set_visible(self, visible: bool):
        if self.current_mode == GSplatMode.NORMAL:
            self.gsplat.visible = visible
        elif self.current_mode == GSplatMode.POINTS_VIEW:
            self.gsplat_points_view.visible = visible

    # 外部から visible を呼び出すときはこちら（チェックボックスの値も同期する）
    def set_visible(self, visible: bool):
        self._set_visible(visible)
        if self.visible_checkbox is not None:
            self.visible_checkbox.value = visible

    # 渡されたチェックボックスを visible と結びつける
    def attachVisibleCheckbox(self, checkbox: viser.GuiCheckboxHandle):
        @checkbox.on_update
        def _(_):
            self._set_visible(checkbox.value)
        self.visible_checkbox = checkbox
        

    
