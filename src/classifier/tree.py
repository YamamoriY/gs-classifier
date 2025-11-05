import numpy as np
import numpy.typing as npt
from src.lib.types import GSplatData
from src.viewer.viewer import Viewer
from src.lib.gsloader import load_ply_file
from pathlib import Path
import matplotlib.pyplot as plt
from src.lib.leaf import rgb_to_hsv
from src.lib.leaf import LeafClassifier

if __name__ == "__main__":
    ground_gs = GSplatData.load_from_npz("data/ground_gs.npz")
    above_ground_gs = GSplatData.load_from_npz("data/above_ground_gs.npz")

    leaf_classifier = LeafClassifier(above_ground_gs)
    leaf_indices = leaf_classifier.classify_leaf()
    above_ground_gs.labels[leaf_indices] = 1
    above_ground_gs.print_shape()
    leaf_gs = above_ground_gs.split_by_label()[1]
    objects_gs = above_ground_gs.split_by_label()[0]
    
    viewer = Viewer()
    viewer.add_gsplat(ground_gs, name="ground", folder_name="ground")
    viewer.add_gsplat(objects_gs, name="objects", folder_name="objects")
    viewer.add_gsplat(leaf_gs, name="leaf", folder_name="leaf")
    viewer.run()
