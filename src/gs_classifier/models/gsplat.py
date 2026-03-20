"""Gaussian Splat のデータクラス定義。"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class GSplatData:
    """3D Gaussian Splatting の基本データクラス。

    ガウシアンの位置、色、不透明度、共分散行列を保持する。
    ラベルによる分割や座標変換、NPZ 形式での永続化をサポートする。

    Attributes:
        centers: ガウシアン中心座標 (N, 3)。
        rgbs: RGB 色 (N, 3)。
        opacities: 不透明度 (N, 1)。
        covariances: 共分散行列 (N, 3, 3)。
        labels: 分類ラベル (N,)。
        additional_data: 追加データの辞書。
    """

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
        labels: npt.NDArray[np.integer] | None = None,
        additional_data: (dict[str, npt.NDArray[np.floating]] | None) = None,
    ) -> None:
        """GSplatData を初期化する。

        Args:
            centers: ガウシアン中心座標 (N, 3)。
            rgbs: RGB 色 (N, 3)。
            opacities: 不透明度 (N, 1)。
            covariances: 共分散行列 (N, 3, 3)。
            labels: 分類ラベル (N,)。None の場合は全て 0。
            additional_data: 追加データの辞書。
        """
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

    def save_to_npz(self, path: str) -> None:
        """NPZ ファイルに保存する。

        Args:
            path: 出力先のファイルパス。
        """
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

    @classmethod
    def load_from_npz(cls, path: str) -> GSplatData:
        """NPZ ファイルから読み込む。

        Args:
            path: 入力ファイルパス。

        Returns:
            読み込んだ GSplatData。
        """
        data = np.load(path)
        centers = data["centers"]
        rgbs = data["rgbs"]
        opacities = data["opacities"]
        covariances = data["covariances"]
        labels = data["labels"]
        additional_data = {}
        for key in data.keys():
            if key.startswith("additional_"):
                additional_data[key[len("additional_") :]] = data[key]
        return cls(
            centers=centers,
            rgbs=rgbs,
            opacities=opacities,
            covariances=covariances,
            labels=labels,
            additional_data=additional_data,
        )

    def copy(self) -> GSplatData:
        """深いコピーを返す。

        Returns:
            コピーされた GSplatData。
        """
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

    def reset_labels(self) -> GSplatData:
        """全ラベルを 0 にリセットする。

        Returns:
            自身（メソッドチェーン用）。
        """
        self.labels = np.zeros_like(self.labels)
        return self

    def split_by_label(self) -> list[GSplatData]:
        """ラベルに従って分割された GSplatData のリストを返す。

        Returns:
            ラベルの昇順で分割された GSplatData のリスト。
        """
        unique_labels = np.unique(self.labels)
        res = []
        for label in unique_labels:
            mask = self.labels == label
            additional_data = {}
            for key, value in self.additional_data.items():
                if len(value) != len(self.centers):
                    print(
                        f"additional data {key} has wrong shape: "
                        f"{value.shape} != {len(self.centers)}"
                    )
                    print(
                        f"additional data {key} is not inherited "
                        f"to new GSplatData"
                    )
                    continue
                additional_data[key] = value[mask]
            res.append(
                GSplatData(
                    centers=self.centers[mask],
                    rgbs=self.rgbs[mask],
                    opacities=self.opacities[mask],
                    covariances=self.covariances[mask],
                    additional_data=additional_data,
                )
            )
        return res

    def concatenate(self, other: GSplatData) -> GSplatData:
        """2 つの GSplatData を結合する。

        Args:
            other: 結合する GSplatData。

        Returns:
            結合された新しい GSplatData。
        """
        return GSplatData(
            centers=np.concatenate([self.centers, other.centers], axis=0),
            rgbs=np.concatenate([self.rgbs, other.rgbs], axis=0),
            opacities=np.concatenate(
                [self.opacities, other.opacities], axis=0
            ),
            covariances=np.concatenate(
                [self.covariances, other.covariances], axis=0
            ),
            labels=np.concatenate([self.labels, other.labels], axis=0),
            additional_data=self.additional_data.copy(),
        )

    def coordinate_transform(self, R: npt.NDArray[np.floating]) -> GSplatData:
        """座標変換を適用する。

        Args:
            R: 3x3 の回転行列。

        Returns:
            自身（メソッドチェーン用）。
        """
        self.centers = self.centers @ R
        self.covariances = np.einsum(
            "ij,njk,kl->nil", R.T, self.covariances, R
        )
        return self

    def print_shape(self) -> None:
        """データの形状とラベル分布を表示する。"""
        print("GSplat Data Shape:")
        print(f"    centers: {self.centers.shape}")
        print(f"    rgbs: {self.rgbs.shape}")
        print(f"    opacities: {self.opacities.shape}")
        print(f"    covariances: {self.covariances.shape}")
        print(
            f"    x range: {np.min(self.centers[:, 0])} "
            f"to {np.max(self.centers[:, 0])}"
        )
        print(
            f"    y range: {np.min(self.centers[:, 1])} "
            f"to {np.max(self.centers[:, 1])}"
        )
        print(
            f"    z range: {np.min(self.centers[:, 2])} "
            f"to {np.max(self.centers[:, 2])}"
        )
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print("    label distribution:")
        for label, count in zip(unique_labels, counts):
            print(
                f"        label {label}: {count} "
                f"({count / len(self.labels) * 100:.2f}%)"
            )
