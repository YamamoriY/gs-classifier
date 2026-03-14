"""viser ベースの 3D ビューアー。"""

import time
from pathlib import Path

import numpy as np
import viser
from PIL import Image

from gs_classifier.models import GSplatData
from gs_classifier.viewer.types import (
    GSplatFolder,
    GSplatHandle,
    GSplatMode,
)


class Viewer:
    """3DGS データを viser で可視化するビューアー。

    GSplatData をラベル別に分割して表示し、
    モード切替や画像出力の GUI を提供する。

    Attributes:
        server: viser サーバーインスタンス。
        gsplatfolders: フォルダ名をキーとした GSplatFolder の辞書。
        folder_gui: ルートの GUI フォルダハンドル。
        image_outs: 画像出力対象のハンドルリスト。
    """

    server: viser.ViserServer
    gsplatfolders: dict[str, GSplatFolder]
    folder_gui: viser.GuiFolderHandle
    image_outs: list[GSplatHandle]

    def __init__(self) -> None:
        """Viewer を初期化し、GUI をセットアップする。"""
        self.server = viser.ViserServer(port=8080)
        self.gsplatfolders = {}
        self.folder_gui = self.server.gui.add_folder("Objects")
        self.image_outs = []
        self.setup()

    def setup(self) -> None:
        """初期設定と GUI コンポーネントを追加する。"""
        # setting
        self.server.scene.set_up_direction((0.0, 0.0, 1.0))  # z方向を上に

        # gui
        button = self.server.gui.add_button("Print Images")

        @button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            # いったん全部オフにする
            for gsplatfolder in self.gsplatfolders.values():
                gsplatfolder.hide_all()
                print(f"hidden gsplatfolder: {gsplatfolder.folder}")
            before = None
            for out_gs in self.image_outs:
                if before is not None:
                    before.set_visible(False)
                out_gs.set_visible(True)
                # 写真を撮る
                mid = np.mean(out_gs.data.centers, axis=0)
                sigma_z = np.std(out_gs.data.centers[:, 2])
                dist = 4.0 * sigma_z  # 係数は適当
                d = [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
                for i in range(len(d)):
                    client.camera.position = (
                        mid[0] + d[i][0] * dist,
                        mid[1] + d[i][1] * dist,
                        mid[2] + d[i][2] * dist,
                    )
                    client.camera.look_at = mid
                    image: np.ndarray = client.get_render(
                        width=600, height=800
                    )
                    output_path = Path(
                        f"tmp/images/{out_gs.gsplat.name}/{i + 1}.png"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(output_path)
                before = out_gs

        mode_dropdown: viser.GuiDropdownHandle = self.server.gui.add_dropdown(
            "Mode",
            options=[
                GSplatMode.NORMAL.value,
                GSplatMode.POINTS_VIEW.value,
                GSplatMode.CLASS_VIEW.value,
            ],
            initial_value=GSplatMode.NORMAL.value,
        )

        @mode_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            for gsplatfolder in self.gsplatfolders.values():
                gsplatfolder.change_mode(GSplatMode(mode_dropdown.value))

    def add_gsplat(
        self,
        gsplatData: GSplatData,
        name: str,
        folder_name: str = "default",
        visible: bool = True,
        image_out: bool = False,
    ) -> None:
        """GSplatData をラベル別に分割して追加する。

        Args:
            gsplatData: 追加する GSplatData。
            name: 表示名のプレフィックス。
            folder_name: 所属フォルダ名。
            visible: 初期表示状態。
            image_out: True の場合、画像出力対象にする。
        """
        labels_set = set(gsplatData.labels)
        for label in labels_set:
            labels_mask = gsplatData.labels == label
            tmp = GSplatData(
                centers=gsplatData.centers[labels_mask],
                rgbs=gsplatData.rgbs[labels_mask],
                opacities=gsplatData.opacities[labels_mask],
                covariances=gsplatData.covariances[labels_mask],
            )
            self._add_gsplat(
                tmp,
                name=f"{name}_{label}",
                folder_name=folder_name,
                visible=visible,
                image_out=image_out,
            )

    def _add_gsplat(
        self,
        gsplatData: GSplatData,
        name: str,
        folder_name: str = "default",
        visible: bool = True,
        image_out: bool = False,
    ) -> None:
        """単一の GSplatData をビューアーに追加する（内部用）。

        Args:
            gsplatData: 追加する GSplatData。
            name: 表示名。
            folder_name: 所属フォルダ名。
            visible: 初期表示状態。
            image_out: True の場合、画像出力対象にする。
        """
        if folder_name not in self.gsplatfolders:
            self._add_folder(folder_name)
        with self.folder_gui:
            with self.gsplatfolders[folder_name].folder:
                checkbox = self.server.gui.add_checkbox(
                    name, initial_value=visible
                )
                gsplat_handle = self.gsplatfolders[folder_name].add_gsplat(
                    GSplatHandle(gsplatData, name, self.server, visible)
                )
                if image_out:
                    self.image_outs.append(gsplat_handle)
                gsplat_handle.attachVisibleCheckbox(checkbox)

    def add_point_cloud(
        self,
        points: np.ndarray,
        name: str,
        colors: np.ndarray | None = None,
    ) -> viser.PointCloudHandle:
        """ポイントクラウドをシーンに追加する。

        Args:
            points: 点群座標 (N, 3)。
            name: 表示名。
            colors: 色の配列 (N, 3)。None の場合は白。

        Returns:
            viser のポイントクラウドハンドル。
        """
        if colors is None:
            colors = np.ones_like(points)
        point_cloud = self.server.scene.add_point_cloud(
            name=f"{name}",
            points=points,
            colors=colors,
            point_size=0.001,
        )
        return point_cloud

    def _add_folder(self, folder_name: str) -> None:
        """GUI フォルダを追加する（内部用）。

        Args:
            folder_name: フォルダ名。
        """
        if folder_name not in self.gsplatfolders:
            with self.folder_gui:
                folder = self.server.gui.add_folder(folder_name)
                gsplatfolder = GSplatFolder(folder)
                self.gsplatfolders[folder_name] = gsplatfolder
                with folder:
                    visibility_buttons = self.server.gui.add_button_group(
                        "Visibility",
                        options=["Show All", "Hide All"],
                    )

                    @visibility_buttons.on_click
                    def _(_: viser.GuiEvent) -> None:
                        if visibility_buttons.value == "Show All":
                            gsplatfolder.show_all()
                        elif visibility_buttons.value == "Hide All":
                            gsplatfolder.hide_all()

                    step_buttons = self.server.gui.add_button_group(
                        "Step", options=["<", ">"]
                    )

                    @step_buttons.on_click
                    def _(_: viser.GuiEvent) -> None:
                        if step_buttons.value == "<":
                            gsplatfolder.show_prev()
                        elif step_buttons.value == ">":
                            gsplatfolder.show_next()

    def run(self) -> None:
        """ビューアーを起動し、無限ループで待機する。"""
        while True:
            time.sleep(10.0)


if __name__ == "__main__":
    from gs_classifier.core import load_ply_file

    viewer = Viewer()
    ply_path = Path(__file__).parent / "../../../data/takino.ply"
    splat_data = load_ply_file(ply_path, center=True)
    splat_data.print_shape()

    # 座標変換（x軸周り-90°）
    R = np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ]
    )
    splat_data.centers = splat_data.centers @ R
    splat_data.covariances = np.einsum(
        "ij,njk,kl->nil", R.T, splat_data.covariances, R
    )

    viewer.add_gsplat(splat_data, name="forest", folder_name="takino")

    # === ここまで本質 ===
    # === ここから複製を追加してるだけ ===

    # ply_path = Path(__file__).parent / "../../../data/cactus.ply"
    # splat_data = load_ply_file(ply_path, center=True)
    # splat_data.print_shape()

    # # 座標変換（x軸周り-90°）
    # R = np.array([
    #     [1, 0, 0],
    #     [0, 0, -1],
    #     [0, 1, 0],
    # ])
    # splat_data.centers = splat_data.centers @ R
    # splat_data.covariances = np.einsum("ij,njk,kl->nil", R.T, splat_data.covariances, R)

    # viewer.add_gsplat(splat_data, name="cactus", folder_name="akan")

    viewer.run()
