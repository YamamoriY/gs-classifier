from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

# 最も基本的な Gaussian Splat のデータクラス
class GSplatData:
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]
    def __init__(
        self,
        centers: npt.NDArray[np.floating],
        rgbs: npt.NDArray[np.floating],
        opacities: npt.NDArray[np.floating],
        covariances: npt.NDArray[np.floating],
    ):
        self.centers = centers
        self.rgbs = rgbs
        self.opacities = opacities
        self.covariances = covariances

    def print_shape(self):
        print(f"GSplat Data Shape:")
        print(f"    centers: {self.centers.shape}")
        print(f"    rgbs: {self.rgbs.shape}")
        print(f"    opacities: {self.opacities.shape}")
        print(f"    covariances: {self.covariances.shape}")
        print(f"    x range: {np.min(self.centers[:, 0])} to {np.max(self.centers[:, 0])}")
        print(f"    y range: {np.min(self.centers[:, 1])} to {np.max(self.centers[:, 1])}")
        print(f"    z range: {np.min(self.centers[:, 2])} to {np.max(self.centers[:, 2])}")


# クラス分類を追加したバージョン
class GSplatDataWithLabels(GSplatData):
    labels: npt.NDArray[np.integer]
    def __init__(
        self,
        centers: npt.NDArray[np.floating],
        rgbs: npt.NDArray[np.floating],
        opacities: npt.NDArray[np.floating],
        covariances: npt.NDArray[np.floating],
        labels: npt.NDArray[np.integer],
    ):
        super().__init__(centers, rgbs, opacities, covariances)
        self.labels = labels

    @classmethod
    def from_gsplat_data(cls, gsplat_data: GSplatData, labels: npt.NDArray[np.integer]) -> GSplatDataWithLabels:
        return cls(
            centers=gsplat_data.centers,
            rgbs=gsplat_data.rgbs,
            opacities=gsplat_data.opacities,
            covariances=gsplat_data.covariances,
            labels=labels,
        )

    def split_by_label(self) -> list[GSplatDataWithLabels]:
        unique_labels = np.unique(self.labels)
        作成中！！



    def print_shape(self):
        super().print_shape()
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"    label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"        label {label}: {count} ({count / len(self.labels) * 100:.2f}%)")