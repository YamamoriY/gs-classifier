from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

# 最も基本的な Gaussian Splat のデータクラス
@dataclass
class GSplatData:
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]

    def print_shape(self):
        print(f"GSplat Data Shape:")
        print(f"  centers: {self.centers.shape}")
        print(f"  rgbs: {self.rgbs.shape}")
        print(f"  opacities: {self.opacities.shape}")
        print(f"  covariances: {self.covariances.shape}")
        print(f"  x range: {np.min(self.centers[:, 0])} to {np.max(self.centers[:, 0])}")
        print(f"  y range: {np.min(self.centers[:, 1])} to {np.max(self.centers[:, 1])}")
        print(f"  z range: {np.min(self.centers[:, 2])} to {np.max(self.centers[:, 2])}")