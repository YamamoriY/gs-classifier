"""単木抽出パイプライン。"""

from pathlib import Path

import numpy as np

from gs_classifier.core import KDTree, save_ply_file
from gs_classifier.models import GSplatData, TrunkLocation
from gs_classifier.viewer import Viewer


def main(no_viewer: bool = False) -> None:
    """幹検出の結果から単木を抽出し、PLY と JSON を出力する。

    tmp/mid_gs.npz, tmp/ground_gs.npz,
    tmp/above_ground_gs.npz が必要。

    Args:
        no_viewer: True の場合、ビューアー表示をスキップ。
    """
    mid_gs = GSplatData.load_from_npz("tmp/mid_gs.npz")
    ground_gs = GSplatData.load_from_npz("tmp/ground_gs.npz")
    above_gs = GSplatData.load_from_npz("tmp/above_ground_gs.npz")

    # 幹位置を算出
    trunk_location = TrunkLocation.from_gs(mid_gs)

    # いったん書き出し用に保存
    save_gs = above_gs.copy()
    # ground_gs と結合
    save_gs = ground_gs.concatenate(save_gs)
    trunk_location_save = TrunkLocation.from_gs(mid_gs)
    R = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ]
    )
    save_gs.coordinate_transform(R)
    trunk_location_save.coordinate_transform(R)
    Path("out").mkdir(parents=True, exist_ok=True)
    save_ply_file(Path("out/gs.ply"), save_gs)
    trunk_location_save.save_to_json("out/trunk_location.json")
    exit()
    # ここまで保存用

    if no_viewer:
        return

    # KDTree で各点を最近傍の幹に割り当て
    trunk_kdtree = KDTree(trunk_location.trunk_locations)
    above_gs.reset_labels()
    for i in range(len(above_gs.centers)):
        num_points, indices, distances = trunk_kdtree.knn_search(
            above_gs.centers[i], 1
        )
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
    viewer.add_gsplat(
        above_gs,
        name="above",
        folder_name="above",
        image_out=True,
    )
    viewer.run()


if __name__ == "__main__":
    main()
