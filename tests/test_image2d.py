"""Image2D のユニットテスト"""

import numpy as np

from gs_classifier.core.image2d import Image2D
from gs_classifier.models import GSplatData, TrunkLocation


class TestImage2D:
    def _make_scene(self):
        """2 本の幹と周囲の点を生成"""
        rng = np.random.default_rng(42)
        # 幹位置
        trunk_locs = np.array([[0, 0, 0], [5, 5, 0]], dtype=np.float64)
        trunk_location = TrunkLocation(trunk_locs)

        # 幹 A 周辺の点 (x≈0, y≈0)
        n_a = 20
        pts_a = rng.normal(0, 0.3, (n_a, 3)).astype(np.float64)

        # 幹 B 周辺の点 (x≈5, y≈5)
        n_b = 20
        pts_b = rng.normal(5, 0.3, (n_b, 3)).astype(np.float64)

        # 遠い点 (分類外)
        n_far = 5
        pts_far = rng.uniform(20, 30, (n_far, 3)).astype(np.float64)

        centers = np.vstack([pts_a, pts_b, pts_far])
        n = len(centers)
        gs = GSplatData(
            centers=centers,
            rgbs=np.ones((n, 3), dtype=np.float32),
            opacities=np.ones((n, 1), dtype=np.float32),
            covariances=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        )
        return gs, trunk_location

    def test_segment_trees_returns_list(self):
        gs, trunk_loc = self._make_scene()
        image2d = Image2D(gs, trunk_loc)
        result = image2d.segment_trees(radius=2.0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_segment_trees_does_not_mutate_original(self):
        gs, trunk_loc = self._make_scene()
        original_centers = gs.centers.copy()
        Image2D(gs, trunk_loc).segment_trees(radius=2.0)
        np.testing.assert_array_equal(gs.centers, original_centers)
