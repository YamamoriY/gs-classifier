import open3d as o3d
import numpy as np
from collections import Counter
import laspy

# --- utility ---
def downsample_with_open3d(points, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    res = np.asarray(pcd_downsampled.points)
    print(f"voxel down sampled. {len(points)} -> {len(res)}")
    return res

# --- convert ---
def to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def to_points(pcd):
    return np.asarray(pcd.points)

# --- save & read ---
# Example: save_points_las("points.las", points, {"label": labels})
def save_points_las(path, points: np.ndarray, extra_dims: dict = {}):
    las = laspy.create()
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    for key, value in extra_dims.items():
        las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(value[0])))
        las[key] = value
    las.write(path)

def save_points_npy(path, points: np.ndarray):
    np.save(path, points)

def read_points_las(path):
    las = laspy.read(path)
    return np.vstack((las.x, las.y, las.z)).transpose(), las    # np と las どっちも返す

def read_points_npy(path):
    return np.load(path)

# --- log ---
def log_points(points):
    print(f"points:")
    print(f"    shape: {points.shape}")
    print(f"    max: {points.max()}, min: {points.min()}, mean: {points.mean()}, std: {points.std()}")

def log_labels(labels, top_n=10,sort_by_value=False):
    print(f"labels:")
    print(f"    shape: {labels.shape}")
    count = Counter(labels.tolist())
    if sort_by_value:
        count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    else:
        count = sorted(count.items(), key=lambda x: x[0])
    print(f"    count: {count[:top_n]} ...(top {top_n})")
