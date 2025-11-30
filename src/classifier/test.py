import numpy as np
import pyransac3d as pyrsc
from src.viewer.viewer import Viewer
from src.lib.types.gstype import GSplatData

points = GSplatData.load_from_npz("tmp/mid_gs.npz")

unique_labels = np.unique(points.labels)
points.labels[points.labels > 5] = -1
trunk_gss = points.split_by_label()[1:]
# for trunk_gs in trunk_gss:
#     cylinder = pyrsc.Cylinder()
#     center, axis, radius, inliers = cylinder.fit(trunk_gs.centers, thresh=0.02, maxIteration=200)
#     print(f"center: {center}, axis: {axis}, radius: {radius}, inliers: {len(inliers)} / {len(trunk_gs.centers)}")
#     theta = np.arccos(axis[0]) * 180 / np.pi
#     print(f"theta: {theta}")
#     trunk_gs.labels[inliers] = 1

# for trunk_gs in trunk_gss:
#     line = pyrsc.Line()
#     slope, axis, inliers = line.fit(trunk_gs.centers, thresh=0.15, maxIteration=200)
#     print(f"slope: {slope}, axis: {axis}, inliers: {len(inliers)} / {len(trunk_gs.centers)}")
#     theta = np.arccos(slope[2]) * 180 / np.pi
#     print(f"theta: {theta}")
#     trunk_gs.labels[inliers] = 1

for i, trunk_gs in enumerate(trunk_gss):
    means = np.mean(trunk_gs.centers, axis=0)
    stds = np.std(trunk_gs.centers, axis=0)
    print(f"{i}")
    print(f"means: {means}, stds: {stds}")

viewer = Viewer()
for i, trunk_gs in enumerate(trunk_gss):
    viewer.add_gsplat(trunk_gs, name=f"trunk_gs_{i}", folder_name=f"trunk_gs_{i}")
viewer.run()