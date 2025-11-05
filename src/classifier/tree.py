import numpy as np
import numpy.typing as npt
from src.lib.types import GSplatData
from src.viewer.viewer import Viewer
from src.lib.gsloader import load_ply_file
from pathlib import Path


if __name__ == "__main__":
    ground_gs = GSplatData.load_from_npz("data/ground_gs.npz")
    above_ground_gs = GSplatData.load_from_npz("data/above_ground_gs.npz")

    hags = above_ground_gs.additional_data["hags"]
    above_ground_gs.labels[hags < 1] = 1
    above_ground_gs.labels[(1 <= hags) & (hags < 2)] = 2
    above_ground_gs.labels[(2 <= hags) & (hags < 3)] = 3
    above_ground_gs.labels[(3 <= hags) & (hags < 4)] = 4
    above_ground_gs.labels[4 <= hags] = 5
    

    viewer = Viewer()
    viewer.add_gsplat(ground_gs, name="ground", folder_name="ground")
    viewer.add_gsplat(above_ground_gs, name="objects", folder_name="objects")
    viewer.run()
