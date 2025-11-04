from pathlib import Path
import numpy as np
from src.lib.gsloader import load_ply_file
from src.viewer.viewer import Viewer
from src.classifier.util.pcd import *
from src.viewer.util.gstypes import GSplatDataWithClass
from src.lib.ground import SegGround
import matplotlib.pyplot as plt
import time
if __name__ == "__main__":
    ply_path = Path(__file__).parent / "../../data/akan.ply"
    splat_data = load_ply_file(ply_path, center=True)
    print("load ply file")
    splat_data.print_shape()

    # 座標変換（x軸周り-90°）
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    splat_data.centers = splat_data.centers @ R
    splat_data.covariances = np.einsum("ij,njk,kl->nil", R.T, splat_data.covariances, R)

    print("start seg ground...")
    time_start = time.time()
    seg_ground = SegGround(splat_data.centers)
    root = seg_ground.ground_heights(depth=6)
    print(len(root.results.points))
    print(f"time: {time.time() - time_start} seconds")

    viewer = Viewer()
    viewer.add_gsplat(splat_data, name="field", folder_name="akan")
    viewer.server.scene.add_point_cloud(
        name="ground",
        points=np.array(root.results.points),
        colors=(255, 0, 0),
        point_size=0.05,
    )
    viewer.run()
