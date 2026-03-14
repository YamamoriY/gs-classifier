import numpy as np
import pytest

from gs_classifier.models import GSplatData


@pytest.fixture
def sample_gs(n: int = 100) -> GSplatData:
    """N=100 のダミー GSplatData を生成する"""
    rng = np.random.default_rng(42)
    return GSplatData(
        centers=rng.standard_normal((n, 3)).astype(np.float32),
        rgbs=rng.uniform(0, 1, (n, 3)).astype(np.float32),
        opacities=rng.uniform(0, 1, (n, 1)).astype(np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
    )


@pytest.fixture
def labeled_gs() -> GSplatData:
    """ラベル付き GSplatData (label 0: 3点, label 1: 2点)"""
    return GSplatData(
        centers=np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0], [11, 0, 0]],
            dtype=np.float32,
        ),
        rgbs=np.array(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
            dtype=np.float32,
        ),
        opacities=np.ones((5, 1), dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32), (5, 1, 1)),
        labels=np.array([0, 0, 0, 1, 1]),
    )
