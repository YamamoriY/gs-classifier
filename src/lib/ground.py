from __future__ import annotations
import math
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
from src.lib.kdtree import KDTree
from typing import Callable, dataclass_transform
from dataclasses import dataclass

# 地面を抽出するプログラム
# 四分木的にフィールドの高さを推定
class SegGround:
    points: np.ndarray
    cylinder_kdtree: KDTree
    x_max: float
    x_min: float
    y_max: float
    y_min: float
    z_max: float
    z_min: float

    def __init__(self, points: np.ndarray):
        self.points = points
        self.cylinder_kdtree = KDTree(points)
        self.x_max = np.max(points[:, 0])
        self.x_min = np.min(points[:, 0])
        self.y_max = np.max(points[:, 1])
        self.y_min = np.min(points[:, 1])
        self.z_max = np.max(points[:, 2])
        self.z_min = np.min(points[:, 2])

    def ground_height(self, point: np.ndarray, radius: float, limit: LimitRange) -> float:
        num_points, indices, distances = self.cylinder_kdtree.cylinder_search(point, radius)
        near_points = self.points[indices]
        # ランダムサンプリング（数は適当）
        if len(near_points) > 80000:
            near_points = near_points[np.random.choice(len(near_points), size=80000, replace=False)]
        near_points = near_points[np.where((near_points[:, 2] > limit.z_min) & (near_points[:, 2] < limit.z_max))]
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
        counts, bin_edges = np.histogram(near_points_z, bins=1000)  # 1000は離散化の数、適当
        max_bin_idx = np.argmax(counts)
        max_density_z = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
        return max_density_z

    # 地面の高さを四分木で求める。メッシュ状（未ソート）のデータを返す
    def ground_heights(self, depth: int = 6):
        calc_height_func = lambda point, radius, limit: self.ground_height(point, radius, limit)
        root_point = np.array([(self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2, 0])
        root_limit = LimitRange(self.z_min, self.z_max)
        root_radius = math.sqrt((self.x_max - self.x_min)**2 + (self.y_max - self.y_min)**2) / 2
        root = QuadNode(root_point, root_radius, root_limit, depth, None, calc_height_func)
        root.insert()
        return root

# 地面として許容する高さ
@dataclass
class LimitRange:
    z_min: float
    z_max: float

# 結果の格納
@dataclass
class GroundResult:
    points: list[np.ndarray]

# 四分木のノード
class QuadNode:
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
        results: GroundResult | None,  # 再帰用、外部から呼ぶときはNone
        calc_height_func: Callable[[np.ndarray, float, LimitRange], float],
    ):
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
    
    # 四分木にノードを追加（再帰）
    def insert(self):
        if self.children is not None or self.depth <= 0:
            self.results.points.append(np.array([self.point[0], self.point[1], self.height]))
            return
        self.children = []
        nexts = self.next_coordinates()
        next_diff = self.radius / 2 / math.sqrt(2) * 0.3     # 地面の傾きの許容量
        next_limit = LimitRange(self.height - next_diff, self.height + next_diff)
        for next in nexts:
            child = QuadNode(next, self.radius / 2, next_limit, self.depth - 1, self.results, self.calc_height_func)
            self.children.append(child)
            child.insert()

    # 次のノードの座標
    def next_coordinates(self):
        shift = self.radius / math.sqrt(2) / 2
        res: list[np.ndarray] = []
        res.append(self.point + np.array([shift, shift, 0]))
        res.append(self.point + np.array([-shift, shift, 0]))
        res.append(self.point + np.array([-shift, -shift, 0]))
        res.append(self.point + np.array([shift, -shift, 0]))
        return res

# GroundResult から地面の高さを補間する
class GroundLerp:
    results: GroundResult
    f: RegularGridInterpolator
    def __init__(self, results: GroundResult):
        self.results = results

        # 線形補完関数を作成する
        # GroundResult からいろいろ整形
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
        self.f = RegularGridInterpolator((x_unique, y_unique), values_2d, method='linear', bounds_error=False, fill_value=np.nan)
    
    # 線形補完 points: (N, 2)
    def lerp(self, points: np.ndarray) -> float:
        return self.f(points)

    # 地面かどうかを判定 points: (N, 3)
    def is_ground(self, points: np.ndarray) -> np.ndarray:
        points_xy = points[:, :2]
        points_ground = self.lerp(points_xy)
        labels = np.where(np.isnan(points_ground), False, points[:, 2] < points_ground + 0.1)
        return labels

