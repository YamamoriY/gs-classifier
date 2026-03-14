"""GroundDetector 統合テスト"""

import numpy as np

from gs_classifier.core.ground import GroundDetector
from gs_classifier.core.ground.groundcore import SegGround
from gs_classifier.models import GSplatData


def _make_scene_gs() -> GSplatData:
    """地面(z≈0) + 地上(z≈2) + 地下(z≈-1) のシンプルなシーンを生成"""
    rng = np.random.default_rng(42)
    n = 300

    # 地面 (z ≈ 0): 一番密
    ground = np.zeros((n, 3), dtype=np.float32)
    ground[:, 0] = rng.uniform(-3, 3, n)
    ground[:, 1] = rng.uniform(-3, 3, n)
    ground[:, 2] = rng.normal(0, 0.03, n)

    # 地上 (z ≈ 2)
    above = np.zeros((100, 3), dtype=np.float32)
    above[:, 0] = rng.uniform(-3, 3, 100)
    above[:, 1] = rng.uniform(-3, 3, 100)
    above[:, 2] = rng.normal(2, 0.1, 100)

    # 地下 (z ≈ -1)
    under = np.zeros((50, 3), dtype=np.float32)
    under[:, 0] = rng.uniform(-3, 3, 50)
    under[:, 1] = rng.uniform(-3, 3, 50)
    under[:, 2] = rng.normal(-1, 0.1, 50)

    centers = np.vstack([ground, above, under])
    total = len(centers)
    return GSplatData(
        centers=centers,
        rgbs=np.ones((total, 3), dtype=np.float32),
        opacities=np.ones((total, 1), dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32), (total, 1, 1)),
    )


class TestSegGround:
    def test_ground_heights_returns_results(self):
        scene = _make_scene_gs()
        seg = SegGround(scene.centers)
        root = seg.ground_heights(depth=3)
        assert root.results is not None
        assert len(root.results.points) > 0

    def test_ground_height_near_zero(self):
        """地面が z≈0 なので推定高もおおよそ 0"""
        scene = _make_scene_gs()
        seg = SegGround(scene.centers)
        root = seg.ground_heights(depth=3)
        heights = np.array([p[2] for p in root.results.points])
        assert np.abs(np.median(heights)) < 0.5


class TestGroundDetector:
    def test_run_returns_three_splits(self):
        scene = _make_scene_gs()
        detector = GroundDetector(scene)
        ground_gs, under_gs, above_gs = detector.run(
            under_threshold=0.3, above_threshold=0.5
        )
        assert len(ground_gs.centers) > 0
        assert len(above_gs.centers) > 0
        total = (
            len(ground_gs.centers)
            + len(under_gs.centers)
            + len(above_gs.centers)
        )
        assert total == len(scene.centers)

    def test_hags_stored_in_additional_data(self):
        scene = _make_scene_gs()
        detector = GroundDetector(scene)
        detector.run(under_threshold=0.3, above_threshold=0.5)
        assert "hags" in scene.additional_data
        assert len(scene.additional_data["hags"]) == len(scene.centers)
