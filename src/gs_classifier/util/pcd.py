"""点群データのユーティリティ関数。"""

from collections import Counter

import laspy
import numpy as np
import open3d as o3d


# --- utility ---
def downsample_with_open3d(
    points: np.ndarray, voxel_size: float = 0.01
) -> np.ndarray:
    """Open3D でボクセルダウンサンプリングを行う。

    Args:
        points: 入力点群 (N, 3)。
        voxel_size: ボクセルサイズ。

    Returns:
        ダウンサンプリングされた点群。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    res = np.asarray(pcd_downsampled.points)
    print(f"voxel down sampled. {len(points)} -> {len(res)}")
    return res


# --- convert ---
def to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    """NumPy 配列を Open3D PointCloud に変換する。

    Args:
        points: 点群 (N, 3)。

    Returns:
        Open3D PointCloud。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def to_points(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Open3D PointCloud を NumPy 配列に変換する。

    Args:
        pcd: Open3D PointCloud。

    Returns:
        点群の NumPy 配列 (N, 3)。
    """
    return np.asarray(pcd.points)


# --- save & read ---
# Example: save_points_las("points.las", points, {"label": labels})
def save_points_las(
    path: str,
    points: np.ndarray,
    extra_dims: dict | None = None,
) -> None:
    """点群を LAS ファイルに保存する。

    Args:
        path: 出力先のファイルパス。
        points: 点群 (N, 3)。
        extra_dims: 追加次元の辞書（例: {"label": labels}）。
    """
    if extra_dims is None:
        extra_dims = {}
    las = laspy.create()
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    for key, value in extra_dims.items():
        las.add_extra_dim(
            laspy.ExtraBytesParams(name=key, type=type(value[0]))
        )
        las[key] = value
    las.write(path)


def save_points_npy(path: str, points: np.ndarray) -> None:
    """点群を NPY ファイルに保存する。

    Args:
        path: 出力先のファイルパス。
        points: 点群 (N, 3)。
    """
    np.save(path, points)


def read_points_las(
    path: str,
) -> tuple[np.ndarray, laspy.LasData]:
    """LAS ファイルから点群を読み込む。

    Args:
        path: 入力ファイルパス。

    Returns:
        (点群 (N, 3), LasData オブジェクト) のタプル。
    """
    las = laspy.read(path)
    return np.vstack((las.x, las.y, las.z)).transpose(), las


def read_points_npy(path: str) -> np.ndarray:
    """NPY ファイルから点群を読み込む。

    Args:
        path: 入力ファイルパス。

    Returns:
        点群の NumPy 配列。
    """
    return np.load(path)


# --- log ---
def log_points(points: np.ndarray) -> None:
    """点群の統計情報をログ出力する。

    Args:
        points: 点群。
    """
    print("points:")
    print(f"    shape: {points.shape}")
    print(
        f"    max: {points.max()}, min: {points.min()}, "
        f"mean: {points.mean()}, std: {points.std()}"
    )


def log_labels(
    labels: np.ndarray,
    top_n: int = 10,
    sort_by_value: bool = False,
) -> None:
    """ラベルの分布をログ出力する。

    Args:
        labels: ラベル配列。
        top_n: 表示する上位件数。
        sort_by_value: True の場合、件数でソート。
    """
    print("labels:")
    print(f"    shape: {labels.shape}")
    count = Counter(labels.tolist())
    if sort_by_value:
        count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    else:
        count = sorted(count.items(), key=lambda x: x[0])
    print(f"    count: {count[:top_n]} ...(top {top_n})")
