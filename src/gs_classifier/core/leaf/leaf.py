"""GSplatData に対する葉検出ラッパー。"""

from gs_classifier.core.leaf.leafcore import LeafClassifier
from gs_classifier.models import GSplatData


class LeafDetector:
    """GSplatData から葉を検出し分離する。

    Attributes:
        gs: 処理対象の GSplatData。
    """

    gs: GSplatData

    def __init__(self, gs: GSplatData) -> None:
        """LeafDetector を初期化する。

        Args:
            gs: 処理対象の GSplatData。
        """
        self.gs = gs

    def run(self) -> tuple[GSplatData, GSplatData]:
        """葉検出を実行し、葉とそれ以外に分割する。

        Returns:
            (葉の GSplatData, それ以外の GSplatData) のタプル。
        """
        leaf_classifier = LeafClassifier(self.gs)
        indices = leaf_classifier.classify_leaf()
        self.gs.labels[indices] = 1
        leaf_gs = self.gs.split_by_label()[1]
        objects_gs = self.gs.split_by_label()[0]
        return leaf_gs, objects_gs
