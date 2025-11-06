from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.lib.gsloader import load_ply_file
from src.viewer.viewer import Viewer
from src.classifier.util.pcd import *
from src.lib.types.types import GSplatData
from src.lib.denoise.denoise import NoiseRemover
from src.lib.ground.ground import GroundDetector

# 3DGS の .ply ファイル (postshot 出力は確認済み）を読み込んで、
# ノイズ除去、地面分類、保存を行う

if __name__ == "__main__":
    # === ロード ===
    print("load ply file")
    ply_path = Path(__file__).parent / "../../data/akan.ply"
    splat_data = load_ply_file(ply_path, center=True)

    # 座標変換（x軸周り-90°）
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    splat_data.coordinate_transform(R)

    # === ノイズ除去 ===
    print("start denoise")
    denoise = NoiseRemover(splat_data)
    gs, noise_gs = denoise.denoise_2d_dbscan(radius=0.1)

    # === 地面を作成 ===
    detect_ground = GroundDetector(gs)
    ground_gs, under_ground_gs, above_ground_gs = detect_ground.detect_ground()

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
