# Leaf のラッパー
from src.lib.leaf.leafcore import LeafClassifier
from src.lib.types.gstype import GSplatData

class LeafDetector:
    gs: GSplatData
    def __init__(self, gs: GSplatData):
        self.gs = gs

    def detect_leaf(self) -> tuple[GSplatData, GSplatData]:
        leaf_classifier = LeafClassifier(self.gs)
        indices = leaf_classifier.classify_leaf()
        self.gs.labels[indices] = 1
        leaf_gs = self.gs.split_by_label()[1]
        objects_gs = self.gs.split_by_label()[0]
        return leaf_gs, objects_gs