"""NoiseRemover.run_3d_density のテスト"""

import numpy as np

from gs_classifier.core.denoise import NoiseRemover
from gs_classifier.models import GSplatData


def _make_dense_with_sparse() -> GSplatData:
    """密なクラスタ + 疎な点群を生成"""
    rng = np.random.default_rng(42)
    # 密なクラスタ (500点)
    dense = rng.standard_normal((500, 3)).astype(np.float32) * 0.05
    # 疎なノイズ (20点、広い範囲に散在)
    sparse = rng.uniform(-5, 5, (20, 3)).astype(np.float32)
    centers = np.vstack([dense, sparse])
    n = len(centers)
    return GSplatData(
        centers=centers,
        rgbs=np.ones((n, 3), dtype=np.float32),
        opacities=np.ones((n, 1), dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
    )


class TestNoiseRemoverDensity:
    def test_run_3d_density_splits_correctly(self):
        gs = _make_dense_with_sparse()
        total = len(gs.centers)
        remover = NoiseRemover(gs)
        clean_gs, noise_gs = remover.run_3d_density(radius=0.3, point_count=10)
        assert len(clean_gs.centers) + len(noise_gs.centers) == total

    def test_run_3d_density_keeps_dense_region(self):
        gs = _make_dense_with_sparse()
        remover = NoiseRemover(gs)
        clean_gs, noise_gs = remover.run_3d_density(radius=0.3, point_count=10)
        # 密な領域の大部分は残る
        assert len(clean_gs.centers) >= 400

    def test_run_2d_dbscan_no_data_loss(self):
        """2d_dbscan でデータが欠落しないことを確認"""
        gs = _make_dense_with_sparse()
        total = len(gs.centers)
        remover = NoiseRemover(gs)
        clean_gs, noise_gs = remover.run_2d_dbscan(radius=0.3)
        assert len(clean_gs.centers) + len(noise_gs.centers) == total
