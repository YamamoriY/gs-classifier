from pathlib import Path
import numpy as np
from src.lib.gsloader import load_ply_file
from src.viewer.viewer import Viewer
from src.classifier.util.pcd import *
from src.lib.types import GSplatData
from src.lib.ground import SegGround, GroundLerp
import matplotlib.pyplot as plt
import time
from src.lib.denoise import Denoise

# 3DGS の .ply ファイル (postshot 出力は確認済み）を読み込んで、
# ノイズ除去、地面分類、保存を行う

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
    splat_data.labels[indices] = 1

    # === ノイズを分離 ===
    noise_gs = splat_data.split_by_label()[0]
    gs = splat_data.split_by_label()[1]

    # === 地面を作成 ===
    print("start seg ground")
    seg_ground = SegGround(gs.centers)
    ground = seg_ground.ground_heights(depth=6)

    # === 地面を分類 ===
    ground_lerp = GroundLerp(ground.results)
    under_ground_indices, ground_indices, above_ground_indices, hags = ground_lerp.classify_ground(gs.centers, above_threshold=0.15)
    gs.labels[under_ground_indices] = 0
    gs.labels[ground_indices] = 1
    gs.labels[above_ground_indices] = 2
    gs.additional_data["hags"] = hags
    under_ground_gs = gs.split_by_label()[0]
    ground_gs = gs.split_by_label()[1]
    above_ground_gs = gs.split_by_label()[2]

    # === 保存 ===
    noise_gs.save_to_npz("data/noise_gs.npz")
    ground_gs.save_to_npz("data/ground_gs.npz")
    under_ground_gs.save_to_npz("data/under_ground_gs.npz")
    above_ground_gs.save_to_npz("data/above_ground_gs.npz")

    # === 表示 ===
    viewer = Viewer()
    viewer.add_gsplat(
        ground_gs,
        name="ground",
        folder_name="ground",
    )
    viewer.add_gsplat(
        under_ground_gs,
        name="under_ground",
        folder_name="under_ground",
    )
    viewer.add_gsplat(
        above_ground_gs,
        name="above_ground",
        folder_name="above_ground",
    )
    viewer.add_gsplat(
        noise_gs,
        name="noise",
        folder_name="noise",
    )
    viewer.run()
