import open3d as o3d
import numpy as np

# 円柱状に距離探索するKDTree
class KDTree:
    points: np.ndarray
    points_flat: np.ndarray
    pcd_flat: o3d.geometry.PointCloud
    kdtree_flat: o3d.geometry.KDTreeFlann
    points_height: np.ndarray       # 一応書き足し．エラー未チェックだけど書き間違えとかなければ大丈夫なはず
    pcd_height: o3d.geometry.PointCloud
    kdtree_height: o3d.geometry.KDTreeFlann

    def __init__(self, points):
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

    # 円柱状に距離探索
    def cylinder_search(self, point, radius):
        search_point = np.array([point[0], point[1], 0])
        [num_points, indices, distances] = self.kdtree_flat.search_radius_vector_3d(search_point, radius)
        return num_points, indices, distances

    # 円柱状に最近傍探索
    def knn_search(self, point, k):
        search_point = np.array([point[0], point[1], 0])
        [num_points, indices, distances] = self.kdtree_flat.search_knn_vector_3d(search_point, k)
        return num_points, indices, distances

    # [z-dz, z+dz] の範囲を探索
    def height_search(self, z, dz):
        search_point = np.array([0, 0, z])
        [num_points, indices, distances] = self.kdtree_height.search_radius_vector_3d(search_point, dz)
        return num_points, indices, distances
