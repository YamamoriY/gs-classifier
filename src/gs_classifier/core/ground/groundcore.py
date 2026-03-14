"""地面高さ推定のコアアルゴリズム（四分木ベース）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gs_classifier.core.kdtree import KDTree


class SegGround:
    """四分木で地面の高さを推定する。

    フィールドを四分木で再帰的に分割し、各セルの最頻値高度を
    地面高さとして推定する。

    Attributes:
        points: 3D 点群 (N, 3)。
        cylinder_kdtree: 円柱探索用 KDTree。
    """

    points: np.ndarray
    cylinder_kdtree: KDTree
    x_max: float
    x_min: float
    y_max: float
    y_min: float
    z_max: float
    z_min: float

    def __init__(self, points: np.ndarray) -> None:
        """SegGround を初期化する。

        Args:
            points: 3D 点群 (N, 3)。
        """
        self.points = points
        self.cylinder_kdtree = KDTree(points)
        self.x_max = np.max(points[:, 0])
        self.x_min = np.min(points[:, 0])
        self.y_max = np.max(points[:, 1])
        self.y_min = np.min(points[:, 1])
        self.z_max = np.max(points[:, 2])
        self.z_min = np.min(points[:, 2])

    def ground_height(
        self,
        point: np.ndarray,
        radius: float,
        limit: LimitRange,
    ) -> float:
        """指定点周辺の地面高さをヒストグラムの最頻値から推定する。

        Args:
            point: 探索中心点 (3,)。
            radius: 円柱探索の半径。
            limit: 高さの許容範囲。

        Returns:
            推定された地面高さ。
        """
        num_points, indices, distances = self.cylinder_kdtree.cylinder_search(
            point, radius
        )
        near_points = self.points[indices]
        # ランダムサンプリング（数は適当）
        if len(near_points) > 80000:
            near_points = near_points[
                np.random.choice(len(near_points), size=80000, replace=False)
            ]
        near_points = near_points[
            np.where(
                (near_points[:, 2] > limit.z_min)
                & (near_points[:, 2] < limit.z_max)
            )
        ]
        near_points_z = near_points[:, 2]

        # データポイントが少ないときは前の値を返す
        if len(near_points_z) < 50:
            return (limit.z_min + limit.z_max) / 2

        # 旧アルゴリズム（ガウス分布由来, O(n^2)）
        # kde = gaussian_kde(near_points_z)
        # range = np.linspace(np.min(near_points_z), np.max(near_points_z), 1000)
        # density = kde(range)
        # max_density_z = range[np.argmax(density)]

        # ヒストグラムで最頻値を求める（O(n)）
        # 地面以上に極端な平面の集まりはおそらく発生せず、これでよさげ
        counts, bin_edges = np.histogram(
            near_points_z, bins=1000
        )  # 1000は離散化の数、適当
        max_bin_idx = np.argmax(counts)
        max_density_z = (
            bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]
        ) / 2
        return max_density_z

    def ground_heights(self, depth: int = 6) -> QuadNode:
        """四分木で地面の高さをメッシュ状に推定する。

        Args:
            depth: 四分木の再帰深度。

        Returns:
            ルートの QuadNode（results に推定結果が格納される）。
        """

        def calc_height_func(
            point: np.ndarray, radius: float, limit: LimitRange
        ) -> float:
            return self.ground_height(point, radius, limit)

        root_point = np.array(
            [
                (self.x_max + self.x_min) / 2,
                (self.y_max + self.y_min) / 2,
                0,
            ]
        )
        root_limit = LimitRange(self.z_min, self.z_max)
        root_radius = (
            math.sqrt(
                (self.x_max - self.x_min) ** 2 + (self.y_max - self.y_min) ** 2
            )
            / 2
        )
        root = QuadNode(
            root_point,
            root_radius,
            root_limit,
            depth,
            None,
            calc_height_func,
        )
        root.insert()
        return root


@dataclass
class LimitRange:
    """地面として許容する高さの範囲。

    Attributes:
        z_min: 下限高さ。
        z_max: 上限高さ。
    """

    z_min: float
    z_max: float


@dataclass
class GroundResult:
    """四分木による地面高さ推定の結果。

    Attributes:
        points: 推定された地面点のリスト (x, y, z)。
    """

    points: list[np.ndarray]


class QuadNode:
    """四分木のノード。

    フィールドを 4 分割しながら各セルの地面高さを推定する。

    Attributes:
        point: ノードの中心座標 (3,)。
        height: 推定された地面高さ。
        radius: ノードの探索半径。
        children: 子ノードのリスト。
        results: 全リーフノードの結果を共有する GroundResult。
    """

    point: np.ndarray
    height: float
    radius: float
    children: list[QuadNode] | None
    results: GroundResult | None

    def __init__(
        self,
        point: np.ndarray,
        radius: float,
        limit: LimitRange,
        depth: int,
        results: GroundResult | None,
        calc_height_func: Callable[[np.ndarray, float, LimitRange], float],
    ) -> None:
        """QuadNode を初期化する。

        Args:
            point: ノードの中心座標 (3,)。
            radius: 探索半径。
            limit: 高さの許容範囲。
            depth: 残りの再帰深度。
            results: 結果の共有オブジェクト（外部からは None）。
            calc_height_func: 地面高さを計算するコールバック。
        """
        self.point = point
        self.limit = limit
        self.radius = radius
        self.height = calc_height_func(point, radius, limit)
        self.children = None
        self.depth = depth
        self.calc_height_func = calc_height_func
        if results is None:
            self.results = GroundResult(points=[])
        else:
            self.results = results

    def insert(self) -> None:
        """四分木にノードを再帰的に追加する。"""
        if self.children is not None or self.depth <= 0:
            self.results.points.append(
                np.array([self.point[0], self.point[1], self.height])
            )
            return
        self.children = []
        nexts = self.next_coordinates()
        next_diff = self.radius / 2 / math.sqrt(2) * 0.3  # 地面の傾きの許容量
        next_limit = LimitRange(
            self.height - next_diff, self.height + next_diff
        )
        for next in nexts:
            child = QuadNode(
                next,
                self.radius / 2,
                next_limit,
                self.depth - 1,
                self.results,
                self.calc_height_func,
            )
            self.children.append(child)
            child.insert()

    def next_coordinates(self) -> list[np.ndarray]:
        """子ノードの 4 つの中心座標を計算する。

        Returns:
            4 つの子ノード中心座標のリスト。
        """
        shift = self.radius / math.sqrt(2) / 2
        res: list[np.ndarray] = []
        res.append(self.point + np.array([shift, shift, 0]))
        res.append(self.point + np.array([-shift, shift, 0]))
        res.append(self.point + np.array([-shift, -shift, 0]))
        res.append(self.point + np.array([shift, -shift, 0]))
        return res


class GroundLerp:
    """GroundResult から地面高さを線形補間する。

    Attributes:
        results: 四分木の推定結果。
        f: scipy の補間関数。
    """

    results: GroundResult
    f: RegularGridInterpolator

    def __init__(self, results: GroundResult) -> None:
        """GroundLerp を初期化する。

        Args:
            results: 四分木の推定結果。
        """
        self.results = results

        # 線形補完関数を作成する
        points_array = np.array(results.points)
        x_unique = np.sort(np.unique(points_array[:, 0]))
        y_unique = np.sort(np.unique(points_array[:, 1]))
        nx, ny = len(x_unique), len(y_unique)
        value_dict = {(pt[0], pt[1]): pt[2] for pt in results.points}
        values_2d = np.zeros((nx, ny))
        for i, x in enumerate(x_unique):
            for j, y in enumerate(y_unique):
                values_2d[i, j] = value_dict[(x, y)]

        # 範囲外のときはNaN
        self.f = RegularGridInterpolator(
            (x_unique, y_unique),
            values_2d,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def lerp(self, points: np.ndarray) -> np.ndarray:
        """XY 座標から地面高さを線形補間する。

        Args:
            points: XY 座標の配列 (N, 2)。

        Returns:
            補間された地面高さの配列 (N,)。
        """
        return self.f(points)

    def classify_ground(
        self,
        points: np.ndarray,
        under_threshold: float = 0.1,
        above_threshold: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """点群を地下・地面・地上に分類する。

        Args:
            points: 3D 点群 (N, 3)。
            under_threshold: 地下と判定する閾値。
            above_threshold: 地上と判定する閾値。

        Returns:
            (地下インデックス, 地面インデックス, 地上インデックス,
             地面からの高さ) のタプル。
        """
        points_xy = points[:, :2]
        points_ground = self.lerp(points_xy)
        hags = points[:, 2] - points_ground  # 地面からの高さ
        under_ground_indices = np.where(hags < -under_threshold)[0]
        ground_indices = np.where(
            (hags >= -under_threshold) & (hags < above_threshold)
        )[0]
        above_ground_indices = np.where(hags >= above_threshold)[0]
        return (
            under_ground_indices,
            ground_indices,
            above_ground_indices,
            hags,
        )
