"""幹検出パイプライン。"""

from gs_classifier.core.denoise import NoiseRemover
from gs_classifier.core.leaf import LeafDetector
from gs_classifier.core.trunk import TrunkClassifier
from gs_classifier.models import GSplatData
from gs_classifier.viewer import Viewer


def main(no_viewer: bool = False) -> None:
    """地面抽出の結果から幹を検出する。

    tmp/ground_gs.npz と tmp/above_ground_gs.npz が必要。

    Args:
        no_viewer: True の場合、ビューアー表示をスキップ。
    """
    ground_gs = GSplatData.load_from_npz("tmp/ground_gs.npz")
    above_ground_gs = GSplatData.load_from_npz("tmp/above_ground_gs.npz")

    # 葉を分離
    detect_leaf = LeafDetector(above_ground_gs)
    leaf_gs, objects_gs = detect_leaf.run()

    # 中央高度を抜き出し
    hags = objects_gs.additional_data["hags"]
    objects_gs.labels[(hags > 0.7) & (hags < 1.7)] = 1
    mid_gs = objects_gs.split_by_label()[1]
    else_gs = objects_gs.split_by_label()[0]

    denoise = NoiseRemover(mid_gs)
    mid_gs, noise_gs = denoise.run_3d_density(radius=0.1, point_count=200)

    trunk_detector = TrunkClassifier(mid_gs)
    mid_gs = trunk_detector.run()

    # 保存
    mid_gs.save_to_npz("tmp/mid_gs.npz")
    print("saved to tmp/mid_gs.npz")

    if no_viewer:
        return

    viewer = Viewer()
    viewer.add_gsplat(
        ground_gs,
        name="ground",
        folder_name="ground",
        visible=False,
    )
    viewer.add_gsplat(mid_gs, name="mid_trunk", folder_name="mid_trunk")
    viewer.add_gsplat(
        noise_gs,
        name="mid_noise",
        folder_name="mid_noise",
        visible=False,
    )
    viewer.add_gsplat(
        else_gs,
        name="else",
        folder_name="else",
        visible=False,
    )
    viewer.add_gsplat(
        leaf_gs,
        name="leaf",
        folder_name="leaf",
        visible=False,
    )
    viewer.run()


if __name__ == "__main__":
    main()
