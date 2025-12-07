from src.lib.types.gstype import GSplatData
from src.lib.types.trunk import TrunkLocation
from src.lib.kdtree import KDTree
from src.viewer.viewer import Viewer
from src.lib.image2d.image2d import Image2D
from src.lib.gsloader import save_ply_file
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    mid_gs = GSplatData.load_from_npz("tmp/mid_gs.npz")
    ground_gs = GSplatData.load_from_npz("tmp/ground_gs.npz")
    above_gs = GSplatData.load_from_npz("tmp/above_ground_gs.npz")

    # いったん書き出し用に保存
    trunk_location_save = TrunkLocation.from_gs(mid_gs)
    save_gs = above_gs.copy()
    # ground_gs と結合
    save_gs = ground_gs.concatenate(save_gs)
    R = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ])
    save_gs.coordinate_transform(R)
    trunk_location_save.coordinate_transform(R)
    save_ply_file(Path("out/gs.ply"), save_gs)
    trunk_location_save.save_to_json("out/trunk_location.json")
    exit()
    # ここまで保存用

    trunk_location = TrunkLocation.from_gs(mid_gs)
    trunk_kdtree = KDTree(trunk_location.trunk_locations)
    above_gs.reset_labels()
    for i in range(len(above_gs.centers)):
        num_points, indices, distances = trunk_kdtree.knn_search(above_gs.centers[i], 1)
        if num_points <= 0:
            print("ここにくるのはおかしいぜ")
            continue
        if distances[0] > 1:
            # 幹から離れすぎているので、なし
            continue
        above_gs.labels[i] = indices[0] + 1

    # image2d = Image2D(above_gs, trunk_location)
    # above_gs = image2d.segment_trees(radius=1)

    viewer = Viewer()
    viewer.add_gsplat(ground_gs, name="ground", folder_name="ground")
    viewer.add_gsplat(above_gs, name="above", folder_name="above", image_out=True)
    viewer.run()