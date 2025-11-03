from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt
import viser

from util.utils import ColorCycle

# 最も基本的な Gaussian Splat のデータクラス
@dataclass
class GSplatData:
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

# クラス分類を追加したバージョン
@dataclass
class GSplatDataWithClass(GSplatData):
    class_ids: npt.NDArray[np.integer]

    # NOTE: print 後で作る

# ビュアーの表示モード
class GSplatMode(Enum):
    NORMAL = "normal"
    CLASS_VIEW = "class view"
    POINTS_VIEW = "points view"     

# GSplatData のハンドラー
# 表示/非表示 と 表示モードを管理する
# 一度作成した Splat は後から変えられないっぽいので、表示形式の数だけ Splat を作成している
class GSplatHandle:
    gsplat: viser.GaussianSplatHandle
    gsplat_class_view: viser.GaussianSplatHandle
    gsplat_points_view: viser.PointCloudHandle
    current_mode: GSplatMode
    visible_checkbox: viser.GuiCheckboxHandle | None
    def __init__(self, data: GSplatData, name: str, server: viser.ViserServer):
        self.current_mode = GSplatMode.NORMAL
        self.visible_checkbox = None
        # 通常
        self.gsplat = server.scene.add_gaussian_splats(
            name=f"{name}",
            centers=data.centers,
            rgbs=data.rgbs,
            opacities=data.opacities,
            covariances=data.covariances,
        )
        # クラスビュー用
        rgb_class_view = data.rgbs.copy()
        rgb_class_view[:, :3] = ColorCycle()()[:3]
        self.gsplat_class_view = server.scene.add_gaussian_splats(
            name=f"{name}_class_view",
            centers=data.centers,
            rgbs=rgb_class_view,
            opacities=data.opacities,
            covariances=data.covariances,
        )
        self.gsplat_class_view.visible = False      
        # ポイントビュー用
        self.gsplat_points_view = server.scene.add_point_cloud(
            name=f"{name}_points_view",
            points=data.centers,
            colors=data.rgbs,
            point_size=0.001,
        )
        self.gsplat_points_view.visible = False

    def non_visible_all(self):
        self.gsplat.visible = False
        self.gsplat_points_view.visible = False
        self.gsplat_class_view.visible = False

    def change_mode(self, mode: GSplatMode):
        old_visibility = False
        if self.current_mode == GSplatMode.NORMAL:
            old_visibility = self.gsplat.visible
        elif self.current_mode == GSplatMode.POINTS_VIEW:
            old_visibility = self.gsplat_points_view.visible
        elif self.current_mode == GSplatMode.CLASS_VIEW:
            old_visibility = self.gsplat_class_view.visible
        self.current_mode = mode
        if mode == GSplatMode.NORMAL:
            self.non_visible_all()
            self.gsplat.visible = old_visibility
        elif mode == GSplatMode.POINTS_VIEW:
            self.non_visible_all()
            self.gsplat_points_view.visible = old_visibility
        elif mode == GSplatMode.CLASS_VIEW:
            self.non_visible_all()
            self.gsplat_class_view.visible = old_visibility
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # 内部用
    def _set_visible(self, visible: bool):
        if self.current_mode == GSplatMode.NORMAL:
            self.gsplat.visible = visible
        elif self.current_mode == GSplatMode.POINTS_VIEW:
            self.gsplat_points_view.visible = visible
        elif self.current_mode == GSplatMode.CLASS_VIEW:
            self.gsplat_class_view.visible = visible

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
        
# GSplatHandle をまとめたフォルダ
class GSplatFolder:
    _gsplats: list[GSplatHandle]
    _select_index: int = 0
    folder: viser.GuiFolderHandle
    def __init__(self, folder: viser.GuiFolderHandle, gsplats: list[GSplatHandle] = None):
        self._gsplats = gsplats if gsplats is not None else []
        self._select_index = 0
        self.folder = folder

    def add_gsplat(self, gsplatHandle: GSplatHandle):
        self._gsplats.append(gsplatHandle)
        return gsplatHandle

    def show_all(self):
        for gsplat in self._gsplats:
            gsplat.set_visible(True)

    def hide_all(self):
        for gsplat in self._gsplats:
            gsplat.set_visible(False)

    def show_next(self):
        self.hide_all()
        self._select_index = max(0, min(len(self._gsplats) - 1, self._select_index + 1))
        self._gsplats[self._select_index].set_visible(True)

    def show_prev(self):
        self.hide_all()
        self._select_index = max(0, min(len(self._gsplats) - 1, self._select_index - 1))
        self._gsplats[self._select_index].set_visible(True)

    def change_mode(self, mode: GSplatMode):
        for gsplat in self._gsplats:
            gsplat.change_mode(mode)

    
