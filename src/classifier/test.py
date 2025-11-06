import numpy as np
import pyransac3d as pyrsc
from src.viewer.viewer import Viewer

# 3D点群データの準備 (N x 3の配列)
# 例: 円柱状の点群を生成
theta = np.linspace(0, 2*np.pi, 100)
z = np.linspace(0, 10, 100)
radius = 0.3

points = []
for t in theta:
    for zz in z[:20]:
        x = radius * np.cos(t) + np.random.normal(0, 0.1)
        y = radius * np.sin(t) + np.random.normal(0, 0.1)
        points.append([x, y, zz])
        x2 = radius * np.cos(t) + np.random.normal(0, 0.1) + 1 
        y2 = radius * np.sin(t) + np.random.normal(0, 0.1) + 1
        points.append([x2, y2, zz])

points = np.array(points)

# 円柱フィッティング
cylinder = pyrsc.Cylinder()
center, axis, radius, inliers = cylinder.fit(points, thresh=0.2, maxIteration=2000)

# 結果の表示
print(f"円柱の中心軸上の点: {center}")
print(f"円柱の軸方向ベクトル: {axis}")
print(f"円柱の半径: {radius}")
print(f"インライア数: {len(inliers)} / {len(points)}")

# インライア点群の取得
inlier_points = points[inliers]

viewer = Viewer()
viewer.add_point_cloud(inlier_points, name="inlier_points")
viewer.run()