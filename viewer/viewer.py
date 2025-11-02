import viser
import numpy as np
from viser import GaussianSplatHandle
import time
from pathlib import Path
from loader import load_ply_file
from dataclasses import dataclass

# GSとチェックボックスのセット
class GsplatWithGui:
    gsplat: GaussianSplatHandle
    checkbox: viser.GuiCheckboxHandle
    def __init__(self, gsplat: GaussianSplatHandle, checkbox: viser.GuiCheckboxHandle):
        self.gsplat = gsplat
        self.checkbox = checkbox
        @self.checkbox.on_update
        def _(_):
            self.gsplat.visible = self.checkbox.value

    def setVisible(self, visible: bool):
        self.gsplat.visible = visible
        self.checkbox.value = visible

# GSのリスト, フォルダのセット
class GsplatsWithFolder:
    gsplats: list[GsplatWithGui]
    folder: viser.GuiFolderHandle
    select_index: int = 0
    def __init__(self, gsplats: list[GsplatWithGui], folder: viser.GuiFolderHandle):
        self.gsplats = gsplats
        self.folder = folder
        self.select_index = 0

    def show_all(self):
        for gsplat in self.gsplats:
            gsplat.setVisible(True)

    def hide_all(self):
        for gsplat in self.gsplats:
            gsplat.setVisible(False)

    def show_next(self):
        self.hide_all()
        self.select_index = max(0, min(len(self.gsplats) - 1, self.select_index + 1))
        self.gsplats[self.select_index].setVisible(True)

    def show_prev(self):
        self.hide_all()
        self.select_index = max(0, min(len(self.gsplats) - 1, self.select_index - 1))
        self.gsplats[self.select_index].setVisible(True)

class Viewer:
    server: viser.ViserServer
    gsplatfolders: dict[str, GsplatsWithFolder]
    folder_gui: viser.GuiFolderHandle

    def __init__(self):
        self.server = viser.ViserServer(port=8080)
        self.gsplatfolders = {}
        self.folder_gui = self.server.gui.add_folder("Objects")
        self.setup()

    def setup(self):
        # setting
        self.server.scene.set_up_direction((0.0, 0.0, 1.0))  # z方向を上に

    def add_gsplat(
        self,
        name: str,
        centers: np.ndarray,
        rgbs: np.ndarray,
        opacities: np.ndarray,
        covariances: np.ndarray,
        class_name: str = "default",
    ) -> GaussianSplatHandle:
        gsplat = self.server.scene.add_gaussian_splats(
            name=f"{name}",
            centers=centers,
            rgbs=rgbs,
            opacities=opacities,
            covariances=covariances,
        )
        if class_name not in self.gsplatfolders:
            self.add_folder(class_name)
        with self.folder_gui:
            with self.gsplatfolders[class_name].folder:
                checkbox = self.server.gui.add_checkbox(gsplat.name, initial_value=True)
                gsplatwithgui = GsplatWithGui(gsplat, checkbox)
                self.gsplatfolders[class_name].gsplats.append(gsplatwithgui)

        return gsplat

    def add_folder(self, name: str):
        if name not in self.gsplatfolders:
            with self.folder_gui:
                folder = self.server.gui.add_folder(name)
                gsplatfolder = GsplatsWithFolder([], folder)
                self.gsplatfolders[name] = gsplatfolder
                with folder:
                    show_all = self.server.gui.add_button("Show All")
                    @show_all.on_click
                    def _(_):
                        self.gsplatfolders[name].show_all()
                    hide_all = self.server.gui.add_button("Hide All")
                    @hide_all.on_click
                    def _(_):
                        self.gsplatfolders[name].hide_all()
                    prev = self.server.gui.add_button("Prev")
                    @prev.on_click
                    def _(_):
                        self.gsplatfolders[name].show_prev()
                    next = self.server.gui.add_button("Next")
                    @next.on_click
                    def _(_):
                        self.gsplatfolders[name].show_next()

        return self.gsplatfolders[name].folder

    def run(self):
        while True:
            time.sleep(10.0)

if __name__ == "__main__":
    viewer = Viewer()
    ply_path = Path(__file__).parent / "../data/akan.ply"
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

    viewer.add_gsplat(
        name="/akan",
        centers=splat_data.centers,
        rgbs=splat_data.rgbs,
        opacities=splat_data.opacities,
        covariances=splat_data.covariances,
        class_name="akan",
    )

    ply_path = Path(__file__).parent / "../data/cactus.ply"
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

    viewer.add_gsplat(
        name="/cactus",
        centers=splat_data.centers,
        rgbs=splat_data.rgbs,
        opacities=splat_data.opacities,
        covariances=splat_data.covariances,
        class_name="akan",
    )

    splat_data.centers += np.array([0.0, 2.0, 0.0])
    viewer.add_gsplat(
        name="/cactus2",
        centers=splat_data.centers,
        rgbs=splat_data.rgbs,
        opacities=splat_data.opacities,
        covariances=splat_data.covariances,
        class_name="cactus",
    )
    splat_data.centers += np.array([0.0, 2.0, 0.0])
    viewer.add_gsplat(
        name="/cactus3",
        centers=splat_data.centers,
        rgbs=splat_data.rgbs,
        opacities=splat_data.opacities,
        covariances=splat_data.covariances,
        class_name="cactus",
    )
    viewer.run()
    