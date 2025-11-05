from pathlib import Path
import numpy as np
from src.lib.gsloader import load_ply_file
from src.viewer.viewer import Viewer
from src.classifier.util.pcd import *
from src.lib.types import GSplatDataWithLabels
from src.lib.ground import SegGround, GroundLerp
import matplotlib.pyplot as plt
import time
from src.lib.denoise import Denoise
if __name__ == "__main__":
    ply_path = Path(__file__).parent / "../../data/akan.ply"
    splat_data = load_ply_file(ply_path, center=True)
    print("load ply file")
    # splat_data.print_shape()

    # 座標変換（x軸周り-90°）
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    splat_data.centers = splat_data.centers @ R
    splat_data.covariances = np.einsum("ij,njk,kl->nil", R.T, splat_data.covariances, R)

    denoise = Denoise(splat_data.centers)



    noise_indices = denoise.denoise3(radius=0.1)
    print(noise_indices.shape)

    labels = np.zeros(len(splat_data.centers), dtype=int)
    labels[noise_indices] = 1

    gsplat_data = GSplatDataWithLabels.from_gsplat_data(
        gsplat_data=splat_data,
        labels=labels,
    )
    gsplat_data.print_shape()
    viewer = Viewer()
    viewer.add_gsplats(
        gsplat_data,
        name="denoise",
        folder_name="denoise",
    )
    viewer.run()

    print("start seg ground...")
    time_start = time.time()
    seg_ground = SegGround(splat_data.centers)
    root = seg_ground.ground_heights(depth=6)
    print(len(root.results.points))
    print(f"time: {time.time() - time_start} seconds")

    ground_lerp = GroundLerp(root.results)
    under_ground_indices, ground_indices, above_ground_indices = ground_lerp.classify_ground(splat_data.centers)
    labels = np.zeros(len(splat_data.centers), dtype=int)
    labels[under_ground_indices] = 0
    labels[ground_indices] = 1
    labels[above_ground_indices] = 2
    print("ground_labels.shape: ", labels.shape)

    gsplat_data = GSplatDataWithLabels.from_gsplat_data(
        gsplat_data=splat_data,
        labels=labels,
    )

    viewer = Viewer()

    viewer.add_gsplats(
        gsplat_data,
        name="ground",
        folder_name="akan",
    )

    viewer.run()
