from __future__ import annotations
import math
from laspy.copc import dataclass
import numpy as np
from scipy.stats import gaussian_kde
from src.lib.cylinder_kdtree import KDTree
from typing import Callable, dataclass_transform

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

    # ランダムサンプリングで計算量削減したい
    def ground_height(self, point: np.ndarray, radius: float, limit: LimitRange) -> float:
        num_points, indices, distances = self.cylinder_kdtree.cylinder_search(point, radius)
        near_points = self.points[indices]
        near_points = near_points[np.where((near_points[:, 2] > limit.z_min) & (near_points[:, 2] < limit.z_max))]
        near_points_z = near_points[:, 2]
        kde = gaussian_kde(near_points_z)
        range = np.linspace(np.min(near_points_z), np.max(near_points_z), 1000)
        density = kde(range)
        max_density_z = range[np.argmax(density)]
        return max_density_z

    def ground_heights(self):
        calc_height_func = lambda point, radius, limit: self.ground_height(point, radius, limit)
        root_point = np.array([(self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2, 0])
        root_limit = LimitRange(self.z_min, self.z_max)
        root_radius = max(self.x_max - self.x_min, self.y_max - self.y_min) / 6
        root = QuadNode(root_point, root_radius, root_limit, calc_height_func)
        root.insert()
        return root

@dataclass
class LimitRange:
    z_min: float
    z_max: float

class QuadNode:
    point: np.ndarray
    height: float
    radius: float
    children: list[QuadNode] | None

    def __init__(
        self,
        point: np.ndarray,
        radius: float,
        limit: LimitRange,
        calc_height_func: Callable[[np.ndarray, float, LimitRange], float],
    ):
        self.point = point
        self.limit = limit
        self.radius = radius
        self.height = calc_height_func(point, radius, limit)
        self.children = None
        self.calc_height_func = calc_height_func
    
    def insert(self):
        if self.children is not None:
            return
        self.children = []
        nexts = self.next_coordinates()
        next_diff = self.radius / 2 / math.sqrt(2) * 0.3     # 地面の傾きの許容量
        next_limit = LimitRange(self.height - next_diff, self.height + next_diff)
        for next in nexts:
            child = QuadNode(next, self.radius / 2, next_limit, self.calc_height_func)
            self.children.append(child)

    def next_coordinates(self):
        shift = self.radius / math.sqrt(2) / 2
        res: list[np.ndarray] = []
        res.append(self.point + np.array([shift, shift, 0]))
        res.append(self.point + np.array([-shift, shift, 0]))
        res.append(self.point + np.array([-shift, -shift, 0]))
        res.append(self.point + np.array([shift, -shift, 0]))
        return res

