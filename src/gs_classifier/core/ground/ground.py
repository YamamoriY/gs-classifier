"""GSplatData に対する地面検出ラッパー。"""

from gs_classifier.core.ground.groundcore import GroundLerp, SegGround
from gs_classifier.models import GSplatData


class GroundDetector:
    """GSplatData から地面を検出し、地下・地面・地上に分類する。

    Attributes:
        gs: 処理対象の GSplatData。
    """

    gs: GSplatData

    def __init__(self, gs: GSplatData) -> None:
        """GroundDetector を初期化する。

        Args:
            gs: 処理対象の GSplatData。
        """
        self.gs = gs

    def run(
        self,
        under_threshold: float = 0.1,
        above_threshold: float = 0.2,
    ) -> tuple[GSplatData, GSplatData, GSplatData]:
        """地面検出を実行し、3 つの GSplatData に分割する。

        Args:
            under_threshold: 地面より下と判定する高さの閾値。
            above_threshold: 地面より上と判定する高さの閾値。

        Returns:
            (地面, 地下, 地上) の GSplatData タプル。
        """
        print("start detect ground")
        # 地面を作成
        seg_ground = SegGround(self.gs.centers)
        ground = seg_ground.ground_heights(depth=6)

        # 地面を分類
        ground_lerp = GroundLerp(ground.results)
        under_ground_indices, ground_indices, above_ground_indices, hags = (
            ground_lerp.classify_ground(
                self.gs.centers,
                under_threshold=under_threshold,
                above_threshold=above_threshold,
            )
        )
        self.gs.labels[under_ground_indices] = 0
        self.gs.labels[ground_indices] = 1
        self.gs.labels[above_ground_indices] = 2
        self.gs.additional_data["hags"] = hags
        under_ground_gs = self.gs.split_by_label()[0]
        ground_gs = self.gs.split_by_label()[1]
        above_ground_gs = self.gs.split_by_label()[2]
        return ground_gs, under_ground_gs, above_ground_gs
