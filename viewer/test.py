import time
from pathlib import Path
import numpy as np

import viser
from util.loader import load_ply_file


def main():
    server = viser.ViserServer()

    # settings
    # z方向を上に
    server.scene.set_up_direction((0.0, 0.0, 1.0))

    # add objects
    sphere = server.scene.add_icosphere(
        name="/sphere",
        radius=0.3,
        color=(255, 100, 100),
        position=(0.0, 0.0, 0.0),
    )
    box = server.scene.add_box(
        name="/box",
        dimensions=(0.4, 0.4, 0.4),
        color=(100, 255, 100),
        position=(1.0, 0.0, 0.0),
    )

    # Load .ply file
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

    gsplat = server.scene.add_gaussian_splats(
        name="/gsplat",
        centers=splat_data.centers,
        rgbs=splat_data.rgbs,
        opacities=splat_data.opacities,
        covariances=splat_data.covariances,
    )

    # add gui controls
    sphere_visible = server.gui.add_checkbox("Show sphere", initial_value=True)
    sphere_color = server.gui.add_rgb("Sphere color", initial_value=(255, 100, 100))
    box_height = server.gui.add_slider(
        "Box height", min=-1.0, max=1.0, step=0.1, initial_value=0.0
    )

    # connect gui to objects
    @sphere_visible.on_update
    def _(_):
        sphere.visible = sphere_visible.value

    @sphere_color.on_update
    def _(_):
        sphere.color = sphere_color.value

    @box_height.on_update
    def _(_):
        box.position = (1.0, 0.0, box_height.value)

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()