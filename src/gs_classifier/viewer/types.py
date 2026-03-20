"""ビューアー用のデータクラス定義。"""

from __future__ import annotations

from enum import Enum

import viser

from gs_classifier.models import GSplatData
from gs_classifier.viewer.colors import ColorCycle


class GSplatMode(Enum):
    """ビューアーの表示モード。"""

    NORMAL = "normal"
    CLASS_VIEW = "class view"
    POINTS_VIEW = "points view"


class GSplatHandle:
    """GSplatData のビューアー用ハンドル。

    通常表示、クラス表示、ポイント表示の 3 モードを管理する。
    一度作成した Splat は変更できないため、モードごとに作成する。

    Attributes:
        gsplat: 通常表示用ハンドル。
        gsplat_class_view: クラス表示用ハンドル。
        gsplat_points_view: ポイント表示用ハンドル。
        current_mode: 現在の表示モード。
        visible_checkbox: 紐付いたチェックボックス。
        data: 元の GSplatData。
    """

    gsplat: viser.GaussianSplatHandle
    gsplat_class_view: viser.GaussianSplatHandle
    gsplat_points_view: viser.PointCloudHandle
    current_mode: GSplatMode
    visible_checkbox: viser.GuiCheckboxHandle | None
    data: GSplatData

    def __init__(
        self,
        data: GSplatData,
        name: str,
        server: viser.ViserServer,
        visible: bool = True,
    ) -> None:
        """GSplatHandle を初期化する。

        Args:
            data: 表示する GSplatData。
            name: 表示名。
            server: viser サーバーインスタンス。
            visible: 初期表示状態。
        """
        self.data = data
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
        self.set_visible(visible)

    def non_visible_all(self) -> None:
        """全モードの表示を非表示にする。"""
        self.gsplat.visible = False
        self.gsplat_points_view.visible = False
        self.gsplat_class_view.visible = False

    def change_mode(self, mode: GSplatMode) -> None:
        """表示モードを切り替える。

        Args:
            mode: 新しい表示モード。
        """
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

    def _set_visible(self, visible: bool) -> None:
        """現在のモードの表示状態を設定する（内部用）。

        Args:
            visible: 表示状態。
        """
        if self.current_mode == GSplatMode.NORMAL:
            self.gsplat.visible = visible
        elif self.current_mode == GSplatMode.POINTS_VIEW:
            self.gsplat_points_view.visible = visible
        elif self.current_mode == GSplatMode.CLASS_VIEW:
            self.gsplat_class_view.visible = visible

    def set_visible(self, visible: bool) -> None:
        """表示状態を設定し、チェックボックスも同期する。

        Args:
            visible: 表示状態。
        """
        self._set_visible(visible)
        if self.visible_checkbox is not None:
            self.visible_checkbox.value = visible

    def attachVisibleCheckbox(self, checkbox: viser.GuiCheckboxHandle) -> None:
        """チェックボックスと表示状態を紐付ける。

        Args:
            checkbox: 紐付ける GUI チェックボックス。
        """

        @checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._set_visible(checkbox.value)

        self.visible_checkbox = checkbox


class GSplatFolder:
    """GSplatHandle をまとめるフォルダ。

    フォルダ内のハンドルの一括表示/非表示やステップ移動を管理する。

    Attributes:
        folder: viser の GUI フォルダハンドル。
    """

    _gsplats: list[GSplatHandle]
    _select_index: int = 0
    folder: viser.GuiFolderHandle

    def __init__(
        self,
        folder: viser.GuiFolderHandle,
        gsplats: list[GSplatHandle] | None = None,
    ) -> None:
        """GSplatFolder を初期化する。

        Args:
            folder: viser の GUI フォルダハンドル。
            gsplats: 初期ハンドルのリスト。
        """
        self._gsplats = gsplats if gsplats is not None else []
        self._select_index = 0
        self.folder = folder

    def add_gsplat(self, gsplatHandle: GSplatHandle) -> GSplatHandle:
        """ハンドルをフォルダに追加する。

        Args:
            gsplatHandle: 追加するハンドル。

        Returns:
            追加されたハンドル。
        """
        self._gsplats.append(gsplatHandle)
        return gsplatHandle

    def show_all(self) -> None:
        """全ハンドルを表示する。"""
        for gsplat in self._gsplats:
            gsplat.set_visible(True)

    def hide_all(self) -> None:
        """全ハンドルを非表示にする。"""
        for gsplat in self._gsplats:
            gsplat.set_visible(False)

    def show_next(self) -> None:
        """次のハンドルのみ表示する。"""
        self.hide_all()
        self._select_index = max(
            0, min(len(self._gsplats) - 1, self._select_index + 1)
        )
        self._gsplats[self._select_index].set_visible(True)

    def show_prev(self) -> None:
        """前のハンドルのみ表示する。"""
        self.hide_all()
        self._select_index = max(
            0, min(len(self._gsplats) - 1, self._select_index - 1)
        )
        self._gsplats[self._select_index].set_visible(True)

    def change_mode(self, mode: GSplatMode) -> None:
        """全ハンドルの表示モードを変更する。

        Args:
            mode: 新しい表示モード。
        """
        for gsplat in self._gsplats:
            gsplat.change_mode(mode)
