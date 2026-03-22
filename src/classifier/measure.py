from src.lib.types.gstype import GSplatData
from src.lib.trunk.trunk_measure import compute_tree_heights, compute_dbh
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    full_gs = GSplatData.load_from_npz("tmp/forest_labeled_keiteki.npz")
    trunk_only = GSplatData.load_from_npz("tmp/trunk_only_keiteki.npz")

    # 樹高
    heights = compute_tree_heights(full_gs)
    print("=== Tree Heights ===")
    for k, v in heights.items():
        print(f"Tree {k}: {v:.2f} m")

    # DBH
    dbh_results = compute_dbh(trunk_only)
    print("=== DBH ===")
    for k, v in dbh_results.items():
        print(f"Tree {k}: {v.dbh:.3f} m")

    # 円フィット可視化
    n = len(dbh_results)
    if n > 0:
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
