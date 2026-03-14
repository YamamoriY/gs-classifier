"""core モジュールのユニットテスト"""

import numpy as np

from gs_classifier.models import GSplatData


class TestKDTree:
    def test_cylinder_search(self):
        from gs_classifier.core.kdtree import KDTree

        points = np.array(
            [
                [0, 0, 0],
                [0.05, 0, 0],
                [0, 0.05, 0],
                [10, 10, 0],  # 遠い点
            ],
            dtype=np.float64,
        )
        kdtree = KDTree(points)
        num, indices, dists = kdtree.cylinder_search(
            np.array([0, 0, 0]), radius=0.1
        )
        assert num >= 3  # [0,0,0], [0.05,0,0], [0,0.05,0] が含まれる
        assert 3 not in list(indices)[:num]  # 遠い点は含まれない

    def test_cylinder_search_ignores_z(self):
        from gs_classifier.core.kdtree import KDTree

        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 100],  # z だけ遠い
            ],
            dtype=np.float64,
        )
        kdtree = KDTree(points)
        num, indices, dists = kdtree.cylinder_search(
            np.array([0, 0, 50]), radius=0.1
        )
        assert num == 2  # z は無視される

    def test_knn_search(self):
        from gs_classifier.core.kdtree import KDTree

        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
            ],
            dtype=np.float64,
        )
        kdtree = KDTree(points)
        num, indices, dists = kdtree.knn_search(np.array([0.1, 0, 0]), k=2)
        assert num == 2
        assert indices[0] == 0  # 最近傍

    def test_height_search(self):
        from gs_classifier.core.kdtree import KDTree

        points = np.array(
            [
                [0, 0, 1.0],
                [0, 0, 2.0],
                [0, 0, 5.0],
            ],
            dtype=np.float64,
        )
        kdtree = KDTree(points)
        num, indices, dists = kdtree.height_search(z=1.5, dz=0.6)
        assert num == 2  # z=1.0 と z=2.0 が範囲内


class TestDenoiseCore:
    def _make_cluster_data(self):
        """大きなクラスタ + 散在するノイズ点"""
        rng = np.random.default_rng(42)
        cluster = rng.standard_normal((500, 3)) * 0.1  # 密なクラスタ
        noise = rng.uniform(-10, 10, (50, 3))  # 散在するノイズ
        return np.vstack([cluster, noise])

    def test_denoise_dbscan_returns_largest_cluster(self):
        from gs_classifier.core.denoise.denoisecore import DenoiseCore

        points = self._make_cluster_data()
        core = DenoiseCore(points)
        indices = core.denoise_dbscan(radius=0.2)
        # 大きいクラスタの大部分が返される
        assert len(indices) >= 400
        assert len(indices) <= 550


class TestNoiseRemover:
    def test_run_2d_dbscan(self):
        from gs_classifier.core.denoise import NoiseRemover

        rng = np.random.default_rng(42)
        n = 500
        centers = np.zeros((n, 3), dtype=np.float32)
        centers[:, 0] = rng.standard_normal(n) * 0.1
        centers[:, 1] = rng.standard_normal(n) * 0.1
        centers[:, 2] = rng.standard_normal(n) * 0.1
        # ノイズ点を追加
        noise_centers = rng.uniform(-10, 10, (30, 3)).astype(np.float32)
        all_centers = np.vstack([centers, noise_centers])

        gs = GSplatData(
            centers=all_centers,
            rgbs=np.ones((len(all_centers), 3), dtype=np.float32),
            opacities=np.ones((len(all_centers), 1), dtype=np.float32),
            covariances=np.tile(
                np.eye(3, dtype=np.float32), (len(all_centers), 1, 1)
            ),
        )
        remover = NoiseRemover(gs)
        clean_gs, noise_gs = remover.run_2d_dbscan(radius=0.3)
        assert len(clean_gs.centers) > 0
        assert len(noise_gs.centers) > 0
        assert len(clean_gs.centers) + len(noise_gs.centers) == len(
            all_centers
        )


class TestLeafClassifier:
    def test_green_detected_as_leaf(self):
        from gs_classifier.core.leaf.leafcore import LeafClassifier

        # 明確な緑色 (sigmoid 前の値を渡すので、大きい正の値 = 高い値)
        gs = GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            rgbs=np.array(
                [
                    [-2, 2, -2],  # 緑 (sigmoid 後: 低R, 高G, 低B)
                    [2, -2, -2],  # 赤 (sigmoid 後: 高R, 低G, 低B)
                    [-2, -2, 2],  # 青 (sigmoid 後: 低R, 低G, 高B)
                ],
                dtype=np.float32,
            ),
            opacities=np.ones((3, 1), dtype=np.float32),
            covariances=np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)),
        )
        classifier = LeafClassifier(gs)
        indices = classifier.classify_leaf()
        assert 0 in indices  # 緑は葉として検出
        assert 1 not in indices  # 赤は葉ではない

    def test_leaf_detector_split(self):
        from gs_classifier.core.leaf import LeafDetector

        gs = GSplatData(
            centers=np.zeros((4, 3), dtype=np.float32),
            rgbs=np.array(
                [
                    [-2, 2, -2],  # 緑 → 葉
                    [-2, 2, -2],  # 緑 → 葉
                    [2, -2, -2],  # 赤 → 非葉
                    [0, 0, 0],  # グレー → 非葉
                ],
                dtype=np.float32,
            ),
            opacities=np.ones((4, 1), dtype=np.float32),
            covariances=np.tile(np.eye(3, dtype=np.float32), (4, 1, 1)),
        )
        detector = LeafDetector(gs)
        leaf_gs, objects_gs = detector.run()
        assert len(leaf_gs.centers) + len(objects_gs.centers) == 4


class TestGroundCore:
    def test_ground_lerp_classify(self):
        from gs_classifier.core.ground.groundcore import (
            GroundLerp,
            GroundResult,
        )

        # 平坦な地面 (z=0) をシミュレート
        grid_points = []
        for x in np.linspace(-1, 1, 5):
            for y in np.linspace(-1, 1, 5):
                grid_points.append(np.array([x, y, 0.0]))

        result = GroundResult(points=grid_points)
        lerp = GroundLerp(result)

        # 地面付近 / 地上 / 地下の点
        test_points = np.array(
            [
                [0, 0, 0.0],  # 地面
                [0, 0, 1.0],  # 地上
                [0, 0, -0.5],  # 地下
            ]
        )
        under, ground, above, hags = lerp.classify_ground(
            test_points, under_threshold=0.1, above_threshold=0.2
        )
        assert 0 in ground  # z=0 は地面
        assert 1 in above  # z=1 は地上
        assert 2 in under  # z=-0.5 は地下


class TestColorCycle:
    def test_returns_different_colors(self):
        from gs_classifier.viewer.colors import ColorCycle

        cc = ColorCycle()
        cc.reset()
        c1 = cc()
        c2 = cc()
        assert not np.array_equal(c1, c2)

    def test_has_4_components(self):
        from gs_classifier.viewer.colors import ColorCycle

        cc = ColorCycle()
        cc.reset()
        color = cc()
        assert len(color) == 4  # RGBA
