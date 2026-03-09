from src.lib.types.gstype import GSplatData
from src.lib.trunk.trunk_measure import run_pipeline
import numpy as np
import matplotlib.pyplot as plt
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

    full_gs = GSplatData.load_from_npz("tmp/above_ground_gs.npz")
    trunk_slice = GSplatData.load_from_npz("tmp/mid_gs.npz")

    full_gs, heights, dbh_results = run_pipeline(
        full_gs,
        trunk_slice,
    )

    print("=== Tree Heights ===")
    for k, v in heights.items():
        print(f"Tree {k}: {v:.2f} m")

    print("=== DBH ===")
    for k, v in dbh_results.items():
        print(f"Tree {k}: {v.dbh:.3f} m")

    # 円フィット結果をmatplotlibで確認
    n = len(dbh_results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    t = np.linspace(0, 2 * np.pi, 128)
    for ax, (label, r) in zip(axes.flat, dbh_results.items()):
        ax.scatter(r.slice_xy[:, 0], r.slice_xy[:, 1], s=1, alpha=0.5)
        ax.plot(r.center[0] + r.radius * np.cos(t),
                r.center[1] + r.radius * np.sin(t), 'r-', linewidth=2)
        ax.plot(r.center[0], r.center[1], 'r+', markersize=10)
        ax.set_title(f"Tree {label}  DBH={r.dbh:.3f}m")
        ax.set_aspect('equal')
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig("tmp/dbh_circle_fit.png", dpi=150)
    print("saved: tmp/dbh_circle_fit.png")
    plt.close()

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



