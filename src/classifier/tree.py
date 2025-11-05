from src.lib.types import GSplatData
from src.viewer.viewer import Viewer

if __name__ == "__main__":
    ground_gs = GSplatData.load_from_npz("data/ground_gs.npz")
    above_ground_gs = GSplatData.load_from_npz("data/above_ground_gs.npz")

    viewer = Viewer()
    viewer.add_gsplat(ground_gs, name="ground", folder_name="ground")
    viewer.add_gsplat(above_ground_gs, name="objects", folder_name="objects")
    viewer.run()
