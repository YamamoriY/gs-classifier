from src.lib.types.gstype import GSplatData
from src.lib.trunk.trunk_measure import run_pipeline
import numpy as np
from src.viewer.viewer import Viewer

def colorize_by_label(gs: GSplatData):
    labels = gs.labels
    unique_labels = np.unique(labels)

    colors = np.zeros((len(labels), 3))

    rng = np.random.default_rng(0)

    for label in unique_labels:
        if label == -1:
            colors[labels == label] = np.array([0.5, 0.5, 0.5])  # 未分類はグレー
        else:
            color = rng.random(3)
            colors[labels == label] = color

    gs.rgbs = colors
    return gs

def main():

    full_gs = GSplatData.load_from_npz("tmp/above_ground_gs_akan.npz")
    trunk_slice = GSplatData.load_from_npz("tmp/mid_gs_akan.npz")

    ground_z = full_gs.centers[:, 2].min()

    full_gs, heights, dbh = run_pipeline(
        full_gs,
        trunk_slice,
        ground_z
    )

    print("=== Tree Heights ===")
    for k, v in heights.items():
        print(f"Tree {k}: {v:.2f} m")

    print("=== DBH ===")
    for k, v in dbh.items():
        print(f"Tree {k}: {v:.3f} m")

    print("unique labels:", np.unique(full_gs.labels))
    print("label count:", len(np.unique(full_gs.labels)))

    full_gs.save_to_npz("tmp/forest_labeled.npz")



    viewer = Viewer()

    colored_gs = colorize_by_label(full_gs)

    viewer.add_gsplat(
        colored_gs,
        name="labeled_forest",
        folder_name="result",
        image_out=True
    )

    viewer.run()


if __name__ == "__main__":
    main()



