from src.lib.types.gstype import GSplatData
from src.lib.types.trunk import TrunkLocation
from src.lib.kdtree import KDTree
import numpy as np
from src.lib.types.tree import Trees
import matplotlib.pyplot as plt
from src.viewer.viewer import Viewer

# DBH（胸高直径）を推定
class DBHAnalyzer:
    gs_breast: GSplatData
    kdtree_breast: KDTree

    # gs には hags（地面からの高さ） が入っている想定
    # dz = inf にすると全ての点で推定もできる
    def __init__(self, gs: GSplatData, breast_height: float = 1.3, dz: float = 0.1):
        gs_copy = gs.copy()
        gs_copy.reset_labels()
        if "hags" not in gs_copy.additional_data:
            print("WARNING: hags が入っていないので，胸高を抽出できません．全ての点を胸高として扱います．")
            self.gs_breast = gs_copy
            self.kdtree_breast = KDTree(self.gs_breast.centers)
        else:
            gs_copy.labels[(gs_copy.additional_data["hags"] > breast_height - dz) & (gs_copy.additional_data["hags"] < breast_height + dz)] = 1
            self.gs_breast = gs_copy.split_by_label()[1]
            self.kdtree_breast = KDTree(self.gs_breast.centers)

    
    def run(self, trees: Trees, max_dbh: float = 1.0) -> Trees:
        for tree in trees.trees:
            num_points, indices, distances = self.kdtree_breast.cylinder_search(tree.location, max_dbh)
            if num_points <= 0:
                print("Error ? (DBHAnalyzer.run)")
                continue
            points = self.gs_breast.centers[indices]
            # ヒストグラムで最頻値を求める
            histogram = np.histogram(distances, bins=100)
            # 視覚化
            print(f"mean: {np.mean(distances)}")
            print(f"std: {np.std(distances)}")
            print(f"max: {np.max(distances)}")
            print(f"min: {np.min(distances)}")
            print(f"median: {np.median(distances)}")
            # 2つのサブプロットを作成
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # 左側：ヒストグラム
            ax1.hist(distances, bins=100)
            ax1.set_xlabel('Distance')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distance Histogram')
            # 右側：x, y座標の2次元プロット
            ax2.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
            ax2.scatter(tree.location[0], tree.location[1], s=100, c='red', marker='x', label='Tree location')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Points (X-Y view)')
            ax2.set_aspect('equal')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            most_frequent_bin_idx = histogram[0].argmax()
            most_frequent_value = (histogram[1][most_frequent_bin_idx] + histogram[1][most_frequent_bin_idx + 1]) / 2
            tree.dbh = most_frequent_value
        return trees