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
    labels: npt.NDArray[np.integer]   # 指定されなければ全部 0 の配列
    def __init__(
        self,
        centers: npt.NDArray[np.floating],
        rgbs: npt.NDArray[np.floating],
        opacities: npt.NDArray[np.floating],
        covariances: npt.NDArray[np.floating],
        labels: npt.NDArray[np.integer] | None = None,   # optional
    ):
        self.centers = centers
        self.rgbs = rgbs
        self.opacities = opacities
        self.covariances = covariances
        if labels is None:
            labels = np.zeros(len(centers), dtype=int)
        self.labels = labels

    # path に保存
    def save_to_npz(self, path: str):
        np.savez(
            path,
            centers=self.centers,
            rgbs=self.rgbs,
            opacities=self.opacities,
            covariances=self.covariances,
            labels=self.labels,
        )

    # path から読み込む
    @classmethod
    def load_from_npz(cls, path: str) -> GSplatData:
        data = np.load(path)
        return cls(
            centers=data["centers"],
            rgbs=data["rgbs"],
            opacities=data["opacities"],
            covariances=data["covariances"],
            labels=data["labels"],
        )

    def print_shape(self):
        print(f"GSplat Data Shape:")
        print(f"    centers: {self.centers.shape}")
        print(f"    rgbs: {self.rgbs.shape}")
        print(f"    opacities: {self.opacities.shape}")
        print(f"    covariances: {self.covariances.shape}")
        print(f"    x range: {np.min(self.centers[:, 0])} to {np.max(self.centers[:, 0])}")
        print(f"    y range: {np.min(self.centers[:, 1])} to {np.max(self.centers[:, 1])}")
        print(f"    z range: {np.min(self.centers[:, 2])} to {np.max(self.centers[:, 2])}")
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"    label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"        label {label}: {count} ({count / len(self.labels) * 100:.2f}%)")

    # labels に従って分割された GSPlatData を作る。
    # 新たな GSplatData の labels は 0 
    def split_by_label(self) -> list[GSplatData]:
        unique_labels = np.unique(self.labels)
        res = []
        for label in unique_labels:
            mask = self.labels == label
            res.append(GSplatData(
                centers=self.centers[mask],
                rgbs=self.rgbs[mask],
                opacities=self.opacities[mask],
                covariances=self.covariances[mask],
            ))
        return res