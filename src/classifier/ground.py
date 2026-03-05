from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.lib.gsloader import load_ply_file
from src.viewer.viewer import Viewer
from src.classifier.util.pcd import *
from src.lib.types.gstype import GSplatData
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
    theta = -np.pi / 2 / 12
    R2 = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])
    splat_data.coordinate_transform(R)
    splat_data.coordinate_transform(R2)

    # === ノイズ除去 ===
    print("start denoise")
    noise_remover = NoiseRemover(splat_data)
    gs, noise_gs = noise_remover.run_2d_dbscan(radius=0.1)

    # === 地面を作成 ===
    ground_detector = GroundDetector(gs)
    ground_gs, under_ground_gs, above_ground_gs = ground_detector.run(under_threshold=0.1, above_threshold=0.2)

    # === 保存 ===
    noise_gs.save_to_npz("tmp/noise_gs.npz")
    ground_gs.save_to_npz("tmp/ground_gs.npz")
    under_ground_gs.save_to_npz("tmp/under_ground_gs.npz")
    above_ground_gs.save_to_npz("tmp/above_ground_gs.npz")

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
