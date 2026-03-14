"""gsloader の PLY 保存/読み込み round-trip テスト"""

import tempfile
from pathlib import Path

import numpy as np

from gs_classifier.core.gsloader import load_ply_file, save_ply_file
from gs_classifier.models import GSplatData


def _make_gs(n: int = 50) -> GSplatData:
    """正定値共分散行列を持つダミーデータ"""
    rng = np.random.default_rng(123)
    centers = rng.standard_normal((n, 3)).astype(np.float32)
    rgbs = rng.uniform(0.1, 0.9, (n, 3)).astype(np.float32)
    opacities = rng.uniform(0.1, 0.9, (n, 1)).astype(np.float32)
    # 正定値行列を作る: A @ A.T
    A = rng.standard_normal((n, 3, 3)).astype(np.float32) * 0.01
    covariances = np.einsum("nij,nkj->nik", A, A) + np.eye(3)[None] * 1e-4
    return GSplatData(
        centers=centers,
        rgbs=rgbs,
        opacities=opacities,
        covariances=covariances,
    )


class TestPlyRoundTrip:
    def test_save_and_load_preserves_centers(self):
        gs = _make_gs()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ply"
            save_ply_file(path, gs)
            loaded = load_ply_file(path)

            np.testing.assert_allclose(loaded.centers, gs.centers, atol=1e-4)

    def test_save_and_load_preserves_shape(self):
        gs = _make_gs(n=30)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ply"
            save_ply_file(path, gs)
            loaded = load_ply_file(path)

            assert loaded.centers.shape == gs.centers.shape
            assert loaded.rgbs.shape == gs.rgbs.shape
            assert loaded.opacities.shape == gs.opacities.shape
            assert loaded.covariances.shape == gs.covariances.shape

    def test_save_and_load_preserves_colors(self):
        gs = _make_gs()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ply"
            save_ply_file(path, gs)
            loaded = load_ply_file(path)

            np.testing.assert_allclose(loaded.rgbs, gs.rgbs, atol=1e-4)

    def test_save_and_load_preserves_opacities(self):
        gs = _make_gs()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ply"
            save_ply_file(path, gs)
            loaded = load_ply_file(path)

            np.testing.assert_allclose(
                loaded.opacities, gs.opacities, atol=1e-3
            )

    def test_load_with_center_flag(self):
        gs = _make_gs()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ply"
            save_ply_file(path, gs)
            loaded = load_ply_file(path, center=True)

            mean = np.mean(loaded.centers, axis=0)
            np.testing.assert_allclose(mean, [0, 0, 0], atol=1e-5)
