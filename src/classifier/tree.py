import numpy as np
from sklearn.cluster import DBSCAN
from src.lib.types.types import GSplatData
from src.viewer.viewer import Viewer
from src.lib.gsloader import load_ply_file
from pathlib import Path
import matplotlib.pyplot as plt
from src.lib.leaf.leaf import LeafDetector
from src.lib.stem.stem import StemClassifier
from src.lib.kdtree import KDTree
from src.lib.denoise.denoise import NoiseRemover

if __name__ == "__main__":
    ground_gs = GSplatData.load_from_npz("data/ground_gs.npz")
    above_ground_gs = GSplatData.load_from_npz("data/above_ground_gs.npz")

    # 葉を分離
    detect_leaf = LeafDetector(above_ground_gs)
    leaf_gs, objects_gs = detect_leaf.detect_leaf()

    # 中央高度を抜き出し
    hags = objects_gs.additional_data["hags"]
    objects_gs.labels[(hags > 1.5) & (hags < 2.5)] = 1
    mid_gs = objects_gs.split_by_label()[1]
    else_gs = objects_gs.split_by_label()[0]

    denoise = NoiseRemover(mid_gs)
    mid_gs, noise_gs = denoise.denoise_3d_density(radius=0.1, point_count=200)

    stem_detector = StemClassifier(mid_gs)
    mid_gs = stem_detector.dbscan_stem()

    viewer = Viewer()
    viewer.add_gsplat(ground_gs, name="ground", folder_name="ground", visible=False)
    viewer.add_gsplat(mid_gs, name="mid_stem", folder_name="mid_stem")
    viewer.add_gsplat(noise_gs, name="mid_noise", folder_name="mid_noise", visible=False)
    viewer.add_gsplat(else_gs, name="else", folder_name="else", visible=False)
    viewer.add_gsplat(leaf_gs, name="leaf", folder_name="leaf", visible=False)
    viewer.run()
