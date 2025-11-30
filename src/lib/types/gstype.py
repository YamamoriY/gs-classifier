from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

# 最も基本的な Gaussian Splat のデータクラス
class GSplatData:
    # gsplat の必須データ
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]

    # 任意で木の分類などを格納するラベル、指定すれば viewer で分離される
    # 指定されなければ全部 0 の配列
    labels: npt.NDArray[np.integer]

    # 追加データ入れたいときに使う
    # (N, ) の配列を想定。他を入れても動くが、split_by_label で継承されない
    additional_data: dict[str, npt.NDArray[np.floating]] | None = None 

    def __init__(
        self,
        centers: npt.NDArray[np.floating],
        rgbs: npt.NDArray[np.floating],
        opacities: npt.NDArray[np.floating],
        covariances: npt.NDArray[np.floating],
        labels: npt.NDArray[np.integer] | None = None,   # optional
        additional_data: dict[str, npt.NDArray[np.floating]] | None = None,
    ):
        self.centers = centers
        self.rgbs = rgbs
        self.opacities = opacities
        self.covariances = covariances
        if labels is None:
            labels = np.zeros(len(centers), dtype=int)
        self.labels = labels
        if additional_data is None:
            additional_data = {}
        self.additional_data = additional_data

    # path に保存
    def save_to_npz(self, path: str):
        data_dict = {
            "centers": self.centers,
            "rgbs": self.rgbs,
            "opacities": self.opacities,
            "covariances": self.covariances,
            "labels": self.labels,
        }
        if self.additional_data is not None:
            for key, value in self.additional_data.items():
                data_dict[f"additional_{key}"] = value
        np.savez(path, **data_dict)

    # path から読み込む
    @classmethod
    def load_from_npz(cls, path: str) -> GSplatData:
        data = np.load(path)
        centers = data["centers"]
        rgbs = data["rgbs"]
        opacities = data["opacities"]
        covariances = data["covariances"]
        labels = data["labels"]
        additional_data = {}
        for key in data.keys():
            if key.startswith("additional_"):
                additional_data[key[len("additional_"):]] = data[key]
        return cls(centers=centers, rgbs=rgbs, opacities=opacities, covariances=covariances, labels=labels, additional_data=additional_data)

    def copy(self) -> GSplatData:
        res = GSplatData(
            centers=self.centers.copy(),
            rgbs=self.rgbs.copy(),
            opacities=self.opacities.copy(),
            covariances=self.covariances.copy(),
            labels=self.labels.copy(),
        )
        for key, value in self.additional_data.items():
            res.additional_data[key] = value.copy()
        return res

    def reset_labels(self):
        self.labels = np.zeros_like(self.labels)
        return self

    # labels に従って分割された GSPlatData を作る。
    # 新たな GSplatData の labels は 0 
    # 配列の順番は label の小さい順
    def split_by_label(self) -> list[GSplatData]:
        unique_labels = np.unique(self.labels)
        res = []
        for label in unique_labels:
            mask = self.labels == label
            additional_data = {}
            for key, value in self.additional_data.items():
                if len(value) != len(self.centers):
                    print(f"additional data {key} has wrong shape: {value.shape} != {len(self.centers)}")
                    print(f"additional data {key} is not inherited to new GSplatData")
                    continue
                additional_data[key] = value[mask]
            res.append(GSplatData(
                centers=self.centers[mask],
                rgbs=self.rgbs[mask],
                opacities=self.opacities[mask],
                covariances=self.covariances[mask],
                additional_data=additional_data,
            ))
        return res

    def coordinate_transform(self, R: npt.NDArray[np.floating]):
        self.centers = self.centers @ R
        self.covariances = np.einsum("ij,njk,kl->nil", R.T, self.covariances, R)
        return self

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
