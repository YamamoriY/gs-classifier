# Denoise のラッパー
# GSplatData を受け取る
from src.lib.denoise.denoisecore import DenoiseCore
from src.lib.types.gstype import GSplatData
import numpy as np
from src.lib.kdtree import KDTree

class NoiseRemover:
    denoise_core: DenoiseCore
    gs: GSplatData

    def __init__(self, gs: GSplatData):
        self.denoise_core = DenoiseCore(gs.centers)
        self.gs = gs

    # 2次元平面上で DBSCAN して、最大のクラスタを抽出
    def denoise_2d_dbscan(self, radius: float = 0.1) -> tuple[GSplatData, GSplatData]:
        print("start denoise 2d dbscan")
        indices = self.denoise_core.denoise_dbscan(radius)
        self.gs.labels = np.zeros_like(self.gs.labels)
        self.gs.labels[indices] = 1
        noise_gs = self.gs.split_by_label()[0]
        gs = self.gs.split_by_label()[1]
        return gs, noise_gs

    # 密度の低い点を消す
    # NOTE: ここ core と分離するべき
    def denoise_3d_density(self, radius: float = 0.1, point_count: int = 200) -> tuple[GSplatData, GSplatData]:
        print("start denoise 3d density")
        tmp = self.gs.centers.copy()
        tmp[:, 2] = tmp[:, 2] * 0.5
        kdtree = KDTree(tmp)
        for i in range(len(tmp)):
            num_points, indices, distances = kdtree.cylinder_search(tmp[i], radius)
            if num_points < point_count:
                self.gs.labels[i] = 1
        gs = self.gs.split_by_label()[0]
        noise_gs = self.gs.split_by_label()[1]
        return gs, noise_gs