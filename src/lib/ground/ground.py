from src.lib.ground.groundcore import SegGround, GroundLerp
from src.lib.types.gstype import GSplatData

class GroundDetector:
    gs: GSplatData
    def __init__(self, gs: GSplatData):
        self.gs = gs

    def run(self, under_threshold: float = 0.1, above_threshold: float = 0.2) -> tuple[GSplatData, GSplatData, GSplatData]:
        print("start detect ground")
        # 地面を作成
        seg_ground = SegGround(self.gs.centers)
        ground = seg_ground.ground_heights(depth=6)

        # 地面を分類
        ground_lerp = GroundLerp(ground.results)
        under_ground_indices, ground_indices, above_ground_indices, hags = ground_lerp.classify_ground(self.gs.centers, under_threshold=under_threshold, above_threshold=above_threshold)
        self.gs.labels[under_ground_indices] = 0
        self.gs.labels[ground_indices] = 1
        self.gs.labels[above_ground_indices] = 2
        self.gs.additional_data["hags"] = hags
        under_ground_gs = self.gs.split_by_label()[0]
        ground_gs = self.gs.split_by_label()[1]
        above_ground_gs = self.gs.split_by_label()[2]
        return ground_gs, under_ground_gs, above_ground_gs

