from src.lib.types.gstype import GSplatData
from src.lib.trunk.trunk import TrunkClassifier
from src.lib.trunk.trunk_measure import propagate_trunk_ids_minimax, _remove_small_components
import numpy as np
from src.viewer.viewer import Viewer


if __name__ == "__main__":
    full_gs = GSplatData.load_from_npz("tmp/above_ground_gs.npz")
    trunk_slice = GSplatData.load_from_npz("tmp/mid_gs.npz")

    # 1. 幹抽出
    trunk_classifier = TrunkClassifier(trunk_slice)
    trunk_gs = trunk_classifier.run()

    trunk_mask = trunk_gs.labels != -1
    trunk_only = GSplatData(
        centers=trunk_gs.centers[trunk_mask],
        rgbs=trunk_gs.rgbs[trunk_mask],
        opacities=trunk_gs.opacities[trunk_mask],
        covariances=trunk_gs.covariances[trunk_mask],
        labels=trunk_gs.labels[trunk_mask],
        additional_data={k: v[trunk_mask] for k, v in trunk_gs.additional_data.items()},
    )

    # trunk_onlyを保存（計測パイプラインで再利用）
    trunk_only.save_to_npz("tmp/trunk_only_keiteki.npz")
    print("saved: tmp/trunk_only_keiteki.npz")

    # 2. 幹ID伝播
    full_gs, *_ = propagate_trunk_ids_minimax(full_gs, trunk_only, xy_scale=3.5)

    # 3. 孤立小クラスタ除去
    full_gs = _remove_small_components(full_gs)

    print("unique labels:", np.unique(full_gs.labels))
    print("label count:", len(np.unique(full_gs.labels)))

    # 保存
    full_gs.save_to_npz("tmp/forest_labeled_keiteki.npz")
    print("saved: tmp/forest_labeled_keiteki.npz")

    # 可視化
    viewer = Viewer()
    viewer.add_gsplat(full_gs, name="labeled_forest", folder_name="result", image_out=True)
    viewer.run()
