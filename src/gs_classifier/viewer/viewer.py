import viser
import numpy as np
from viser import GaussianSplatHandle
import time
from pathlib import Path
from src.lib.gsloader import load_ply_file
from src.viewer.util.types import GSplatData, GSplatHandle, GSplatFolder, GSplatMode
from src.lib.types.gstype import GSplatData
from PIL import Image

class Viewer:
    server: viser.ViserServer
    gsplatfolders: dict[str, GSplatFolder]
    folder_gui: viser.GuiFolderHandle
    image_outs: list[GSplatHandle]

    def __init__(self):
        self.server = viser.ViserServer(port=8080)
        self.gsplatfolders = {}
        self.folder_gui = self.server.gui.add_folder("Objects")
        self.image_outs = []
        self.setup()

    # 初期設定
    def setup(self):
        # setting
        self.server.scene.set_up_direction((0.0, 0.0, 1.0))  # z方向を上に

        # gui
        button = self.server.gui.add_button("Print Images")
        @button.on_click
        def _(event: viser.GuiEvent):
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
                d = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]
                for i in range(len(d)):
                    client.camera.position = (mid[0] + d[i][0] * dist, mid[1] + d[i][1] * dist, mid[2] + d[i][2] * dist)
                    client.camera.look_at = mid
                    image: np.ndarray = client.get_render(width=600, height=800)
                    output_path = Path(f"tmp/images/{out_gs.gsplat.name}/{i+1}.png")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(output_path)
                before = out_gs

        mode_dropdown: viser.GuiDropdownHandle = self.server.gui.add_dropdown(
            "Mode",
            options=[GSplatMode.NORMAL.value, GSplatMode.POINTS_VIEW.value, GSplatMode.CLASS_VIEW.value],
            initial_value=GSplatMode.NORMAL.value,
        )
        @mode_dropdown.on_update
        def _(_):
            for gsplatfolder in self.gsplatfolders.values():
                gsplatfolder.change_mode(GSplatMode(mode_dropdown.value))

    # gsplatを追加する
    # 同じ folder_name のものは同じフォルダに入る
    # フォルダ内では labels で分類される
    # image_out = True の場合は，ボタンで画像が自動出力できる
    def add_gsplat(
        self,
        gsplatData: GSplatData,
        name: str,
        folder_name: str = "default",
        visible: bool = True,
        image_out: bool = False,
    ) -> GaussianSplatHandle:
        labels_set = set(gsplatData.labels)
        for label in labels_set:
            labels_mask = gsplatData.labels == label
            tmp = GSplatData(
                centers=gsplatData.centers[labels_mask],
                rgbs=gsplatData.rgbs[labels_mask],
                opacities=gsplatData.opacities[labels_mask],
                covariances=gsplatData.covariances[labels_mask],
            )
            self._add_gsplat(tmp, name=f"{name}_{label}", folder_name=folder_name, visible=visible, image_out=image_out)

    # gsplatを追加
    def _add_gsplat(self, gsplatData: GSplatData, name: str, folder_name: str = "default", visible: bool = True, image_out: bool = False):
        if folder_name not in self.gsplatfolders:
            self._add_folder(folder_name)
        with self.folder_gui:
            with self.gsplatfolders[folder_name].folder:
                checkbox = self.server.gui.add_checkbox(name, initial_value=visible)
                gsplat_handle = self.gsplatfolders[folder_name].add_gsplat(GSplatHandle(gsplatData, name, self.server, visible))
                if image_out:
                    self.image_outs.append(gsplat_handle)
                gsplat_handle.attachVisibleCheckbox(checkbox)

    def add_point_cloud(self, points: np.ndarray, name: str, colors: np.ndarray | None = None):
        if colors is None:
            colors = np.ones_like(points)
        point_cloud = self.server.scene.add_point_cloud(
            name=f"{name}",
            points=points,
            colors=colors,
            point_size=0.001,
        )
        return point_cloud

    # フォルダを追加する内部関数
    def _add_folder(self, folder_name: str):
        if folder_name not in self.gsplatfolders:
            with self.folder_gui:
                folder = self.server.gui.add_folder(folder_name)
                gsplatfolder = GSplatFolder(folder)
                self.gsplatfolders[folder_name] = gsplatfolder
                with folder:

                    visibility_buttons = self.server.gui.add_button_group("Visibility", options=["Show All", "Hide All"])
                    @visibility_buttons.on_click
                    def _(_):
                        if visibility_buttons.value == "Show All":
                            gsplatfolder.show_all()
                        elif visibility_buttons.value == "Hide All":
                            gsplatfolder.hide_all()

                    step_buttons = self.server.gui.add_button_group("Step", options=["<", ">"])
                    @step_buttons.on_click
                    def _(_):
                        if step_buttons.value == "<":
                            gsplatfolder.show_prev()
                        elif step_buttons.value == ">":
                            gsplatfolder.show_next()

    # ビュアーを実行
    def run(self):
        while True:
            time.sleep(10.0)

if __name__ == "__main__":
    viewer = Viewer()
    ply_path = Path(__file__).parent / "../../data/takino.ply"
    splat_data = load_ply_file(ply_path, center=True)
    splat_data.print_shape()

    # 座標変換（x軸周り-90°）
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    splat_data.centers = splat_data.centers @ R
    splat_data.covariances = np.einsum("ij,njk,kl->nil", R.T, splat_data.covariances, R)

    viewer.add_gsplat(splat_data, name="forest", folder_name="takino")

    # === ここまで本質 ===
    # === ここから複製を追加してるだけ ===

    # ply_path = Path(__file__).parent / "../../data/cactus.ply"
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

    # splat_data.centers += np.array([0.0, 2.0, 0.0])
    # half_index = len(splat_data.centers) // 2
    # class_ids = np.zeros(len(splat_data.centers), dtype=int)
    # class_ids[:half_index] = 1
    # splat_data.centers[class_ids == 1] += np.array([0.0, 2.0, 0.0])
    # splat_data.rgbs[class_ids == 0] = np.array([1.0, 0.0, 0.0])
    # splat_data.rgbs[class_ids == 1] = np.array([0.0, 1.0, 0.0])
    # viewer.add_gsplats(
    #     gsplatData=GSplatDataWithClass(
    #         centers=splat_data.centers,
    #         rgbs=splat_data.rgbs,
    #         opacities=splat_data.opacities,
    #         covariances=splat_data.covariances,
    #         class_ids=class_ids,
    #     ),
    #     name="classify_cactus",
    #     folder_name="cactus",
    # )


    viewer.run()
    