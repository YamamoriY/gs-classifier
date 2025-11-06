import numpy as np
from sklearn.cluster import DBSCAN
from src.lib.types import GSplatData
from src.viewer.viewer import Viewer
from src.lib.gsloader import load_ply_file
from pathlib import Path
import matplotlib.pyplot as plt
from src.lib.leaf import rgb_to_hsv
from src.lib.leaf import LeafClassifier
from src.lib.stem import StemDetector
from src.lib.kdtree import KDTree



if __name__ == "__main__":
    ground_gs = GSplatData.load_from_npz("data/ground_gs.npz")
    above_ground_gs = GSplatData.load_from_npz("data/above_ground_gs.npz")

    # 葉を分離
    leaf_classifier = LeafClassifier(above_ground_gs)
    leaf_indices = leaf_classifier.classify_leaf()
    above_ground_gs.labels[leaf_indices] = 1
    leaf_gs = above_ground_gs.split_by_label()[1]
    objects_gs = above_ground_gs.split_by_label()[0]

    # 中央高度を抜き出し
    hags = objects_gs.additional_data["hags"]
    objects_gs.labels[(hags > 1.5) & (hags < 2.5)] = 1
    mid_gs = objects_gs.split_by_label()[1]
    else_gs = objects_gs.split_by_label()[0]

    stem_detector = StemDetector(mid_gs)
    mid_gs_stem, mid_gs_noise = stem_detector.dbscan_stem()

    viewer = Viewer()
    viewer.add_gsplat(ground_gs, name="ground", folder_name="ground", visible=False)
    viewer.add_gsplat(mid_gs_stem, name="mid_stem", folder_name="mid_stem")
    viewer.add_gsplat(mid_gs_noise, name="mid_noise", folder_name="mid_noise", visible=False)
    viewer.add_gsplat(else_gs, name="else", folder_name="else", visible=False)
    viewer.add_gsplat(leaf_gs, name="leaf", folder_name="leaf", visible=False)
    viewer.run()
