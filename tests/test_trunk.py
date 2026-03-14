"""TrunkClassifier のユニットテスト"""

import numpy as np

from gs_classifier.core.trunk import TrunkClassifier
from gs_classifier.models import GSplatData


def _make_trunk_gs() -> GSplatData:
    """2本の幹状クラスタ + ノイズを生成する。

    - trunk A: x=0, y=0 付近に z=0~1.5 の縦長クラスタ (200点)
    - trunk B: x=2, y=2 付近に z=0~1.5 の縦長クラスタ (200点)
    - noise:   散在する点 (30点)
    """
    rng = np.random.default_rng(42)
    n_per_trunk = 200

    # 幹 A
    trunk_a = np.zeros((n_per_trunk, 3), dtype=np.float32)
    trunk_a[:, 0] = rng.normal(0, 0.02, n_per_trunk)
    trunk_a[:, 1] = rng.normal(0, 0.02, n_per_trunk)
    trunk_a[:, 2] = rng.uniform(0, 1.5, n_per_trunk)

    # 幹 B
    trunk_b = np.zeros((n_per_trunk, 3), dtype=np.float32)
    trunk_b[:, 0] = rng.normal(2, 0.02, n_per_trunk)
    trunk_b[:, 1] = rng.normal(2, 0.02, n_per_trunk)
    trunk_b[:, 2] = rng.uniform(0, 1.5, n_per_trunk)

    # ノイズ
    noise = rng.uniform(-5, 5, (30, 3)).astype(np.float32)

    centers = np.vstack([trunk_a, trunk_b, noise])
    n = len(centers)
    return GSplatData(
        centers=centers,
        rgbs=np.ones((n, 3), dtype=np.float32),
        opacities=np.ones((n, 1), dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
    )


class TestTrunkClassifier:
    def test_dbscan_trunk_assigns_labels(self):
        gs = _make_trunk_gs()
        tc = TrunkClassifier(gs)
        tc.dbscan_trunk()
        unique = np.unique(gs.labels)
        # ノイズ(-1) + 少なくとも 2 つのクラスタ
        assert len(unique) >= 3

    def test_is_trunk_check_range_removes_short_clusters(self):
        """z 方向の範囲が 0.8 未満のクラスタは除去される"""
        rng = np.random.default_rng(99)
        # 薄いクラスタ (z_range < 0.8)
        n = 100
        centers = np.zeros((n, 3), dtype=np.float32)
        centers[:, 0] = rng.normal(0, 0.02, n)
        centers[:, 1] = rng.normal(0, 0.02, n)
        centers[:, 2] = rng.uniform(0, 0.3, n)  # z_range ≈ 0.3 < 0.8

        gs = GSplatData(
            centers=centers,
            rgbs=np.ones((n, 3), dtype=np.float32),
            opacities=np.ones((n, 1), dtype=np.float32),
            covariances=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            labels=np.zeros(n, dtype=int),  # 全部 label=0
        )
        tc = TrunkClassifier(gs)
        tc.is_trunk_check_range()
        # z_range < 0.8 なので全て -1 に
        assert np.all(gs.labels == -1)

    def test_run_full_pipeline(self):
        gs = _make_trunk_gs()
        tc = TrunkClassifier(gs)
        result = tc.run()
        # ノイズは -1、有効なクラスタはそれ以外
        valid_labels = result.labels[result.labels != -1]
        assert len(valid_labels) > 0
        assert len(np.unique(valid_labels)) >= 1
