"""rgb_to_hsv 関数の直接テストおよび LeafClassifier の追加テスト"""

import numpy as np

from gs_classifier.core.leaf.leafcore import LeafClassifier, rgb_to_hsv
from gs_classifier.models import GSplatData


class TestRgbToHsv:
    def test_output_shape(self):
        rgb = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        hsv = rgb_to_hsv(rgb)
        assert hsv.shape == (2, 3)

    def test_hue_range_0_to_1(self):
        rng = np.random.default_rng(42)
        rgb = rng.uniform(-3, 3, (100, 3)).astype(np.float32)
        hsv = rgb_to_hsv(rgb)
        assert np.all(hsv[:, 0] >= 0)
        assert np.all(hsv[:, 0] <= 1)

    def test_pure_green_hue(self):
        """sigmoid 後に G が支配的な場合、hue は緑の範囲 (0.17~0.67)"""
        # sigmoid(-5)≈0, sigmoid(5)≈1 → 純粋な緑
        rgb = np.array([[-5, 5, -5]], dtype=np.float32)
        hsv = rgb_to_hsv(rgb)
        h = hsv[0, 0]
        assert 0.17 <= h <= 0.67

    def test_pure_red_hue(self):
        rgb = np.array([[5, -5, -5]], dtype=np.float32)
        hsv = rgb_to_hsv(rgb)
        h = hsv[0, 0]
        # 赤は h≈0 or h≈1 の近辺
        assert h < 0.08 or h > 0.9


class TestLeafClassifierEdgeCases:
    def _make_gs(self, rgbs):
        n = len(rgbs)
        return GSplatData(
            centers=np.zeros((n, 3), dtype=np.float32),
            rgbs=np.array(rgbs, dtype=np.float32),
            opacities=np.ones((n, 1), dtype=np.float32),
            covariances=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        )

    def test_all_gray_no_leaf(self):
        """彩度がほぼ 0 のグレーは葉として検出されない"""
        gs = self._make_gs([[0, 0, 0], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
        classifier = LeafClassifier(gs)
        indices = classifier.classify_leaf()
        assert len(indices) == 0

    def test_yellow_detected_as_leaf(self):
        """黄色（高R, 高G, 低B）は葉として検出される"""
        gs = self._make_gs([[3, 2, -5]])  # sigmoid 後: 高R, 高G, 低B → 黄色
        classifier = LeafClassifier(gs)
        indices = classifier.classify_leaf()
        assert 0 in indices
