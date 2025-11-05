import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from src.lib.kdtree import KDTree

# 2次元平面上で密度の低い点をノイズとして除去する

class Denoise:
    points: np.ndarray
    flatten_points: np.ndarray
    def __init__(self, points: np.ndarray):
        self.points = points
        self.kdtree = KDTree(points)

    def denoise(self, radius: float = 0.1):
        noise_indices = []
        for i in range(len(self.points)):
            num_points, indices, distances = self.kdtree.cylinder_search(self.points[i], radius)
            if num_points < 10:
                noise_indices.append(i)
            if i % 100000 == 0:
                print(f"denoise: {i} / {len(self.points)}")
        return np.array(noise_indices)

    # ランダムサンプリングしてノイズを除去する。まぁよさげ
    def denoise2(self, radius: float = 0.1):
        noise_indices = []
        random_points = self.points[np.random.choice(len(self.points), size=1000000, replace=False)]
        for i in range(len(random_points)):
            num_points, indices, distances = self.kdtree.cylinder_search(random_points[i], radius)
            if num_points < 100:
                noise_indices.extend(indices)
            
        return np.unique(noise_indices)

    def denoise3(self, radius: float = 0.1):
        dbscan = DBSCAN(eps=radius * 2, min_samples=50)
        labels = dbscan.fit_predict(self.points)
        unique, counts = np.unique(labels, return_counts=True)
        largest_label = unique[np.argmax(counts)]
        largest_indices = np.where(labels == largest_label)[0]
        return largest_indices