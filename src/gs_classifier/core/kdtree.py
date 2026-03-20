"""円柱状距離探索をサポートする KDTree。"""

import numpy as np
import open3d as o3d


class KDTree:
    """XY 平面投影と高さ方向の KDTree を提供する。

    3D 点群に対して、Z 座標を無視した円柱状の距離探索や
    高さ方向のみの距離探索を行う。

    Attributes:
        points: 元の 3D 点群 (N, 3)。
        points_flat: Z=0 に投影した点群 (N, 3)。
        points_height: X=Y=0 にした高さ方向のみの点群 (N, 3)。
    """

    points: np.ndarray
    points_flat: np.ndarray
    pcd_flat: o3d.geometry.PointCloud
    kdtree_flat: o3d.geometry.KDTreeFlann
    points_height: np.ndarray
    pcd_height: o3d.geometry.PointCloud
    kdtree_height: o3d.geometry.KDTreeFlann

    def __init__(self, points: np.ndarray) -> None:
        """KDTree を初期化する。

        Args:
            points: 3D 点群 (N, 3)。
        """
        self.points = points
        # for cylinder search
        self.points_flat = points.copy()
        self.points_flat[:, 2] = 0
        self.pcd_flat = o3d.geometry.PointCloud()
        self.pcd_flat.points = o3d.utility.Vector3dVector(self.points_flat)
        self.kdtree_flat = o3d.geometry.KDTreeFlann(self.pcd_flat)
        # for height search
        self.points_height = points.copy()
        self.points_height[:, 0] = 0
        self.points_height[:, 1] = 0
        self.pcd_height = o3d.geometry.PointCloud()
        self.pcd_height.points = o3d.utility.Vector3dVector(self.points_height)
        self.kdtree_height = o3d.geometry.KDTreeFlann(self.pcd_height)

    def cylinder_search(
        self, point: np.ndarray, radius: float
    ) -> tuple[int, list[int], list[float]]:
        """XY 平面上で半径内の点を探索する。

        Args:
            point: 探索中心点 (3,)。
            radius: 探索半径。

        Returns:
            (点数, インデックスリスト, 距離リスト) のタプル。
        """
        search_point = np.array([point[0], point[1], 0])
        [num_points, indices, distances] = (
            self.kdtree_flat.search_radius_vector_3d(search_point, radius)
        )
        return num_points, indices, distances

    def knn_search(
        self, point: np.ndarray, k: int
    ) -> tuple[int, list[int], list[float]]:
        """XY 平面上で k 近傍探索を行う。

        Args:
            point: 探索中心点 (3,)。
            k: 近傍点数。

        Returns:
            (点数, インデックスリスト, 距離リスト) のタプル。
        """
        search_point = np.array([point[0], point[1], 0])
        [num_points, indices, distances] = (
            self.kdtree_flat.search_knn_vector_3d(search_point, k)
        )
        return num_points, indices, distances

    def height_search(
        self, z: float, dz: float
    ) -> tuple[int, list[int], list[float]]:
        """高さ方向で [z-dz, z+dz] の範囲にある点を探索する。

        Args:
            z: 探索中心の高さ。
            dz: 探索範囲の半幅。

        Returns:
            (点数, インデックスリスト, 距離リスト) のタプル。
        """
        search_point = np.array([0, 0, z])
        [num_points, indices, distances] = (
            self.kdtree_height.search_radius_vector_3d(search_point, dz)
        )
        return num_points, indices, distances
