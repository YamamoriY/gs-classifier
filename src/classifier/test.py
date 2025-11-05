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
    # === ロード ===
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

    # === ノイズ除去 ===
    print("start denoise")
    denoise = Denoise(splat_data.centers)
    indices = denoise.denoise_dbscan(radius=0.1)
    labels = np.zeros(len(splat_data.centers), dtype=int)
    labels[indices] = 1

    # === ノイズを分離 ===
    gsplat_data = GSplatDataWithLabels.from_gsplat_data(
        gsplat_data=splat_data,
        labels=labels,
    )
    noise_gs = gsplat_data.split_by_label()[0]
    gs = gsplat_data.split_by_label()[1]

    # === 地面を作成 ===
    print("start seg ground")
    seg_ground = SegGround(gs.centers)
    ground = seg_ground.ground_heights(depth=6)

    # === 地面を分類 ===
    ground_lerp = GroundLerp(ground.results)
    under_ground_indices, ground_indices, above_ground_indices = ground_lerp.classify_ground(gs.centers)
    gs.labels[under_ground_indices] = 0
    gs.labels[ground_indices] = 1
    gs.labels[above_ground_indices] = 2

    # === 表示 ===
    viewer = Viewer()
    viewer.add_gsplats(
        gs,
        name="ground",
        folder_name="akan",
    )
    viewer.add_gsplats(
        noise_gs,
        name="noise",
        folder_name="noise",
    )

    viewer.run()
