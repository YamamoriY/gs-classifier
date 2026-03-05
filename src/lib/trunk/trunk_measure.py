from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pyransac3d as pyrsc

from src.lib.types.gstype import GSplatData
from src.lib.trunk.trunk import TrunkClassifier


# ===============================
# 幹ID伝播（Bottom-up Region Growing）
# ===============================
def propagate_trunk_ids_bottom_up(
    full_gs: GSplatData,
    trunk_gs: GSplatData,
    radius: float = 0.5,
    z_step: float = 0.2,
    z_window: float = 0.4,
    seed_radius: float = 0.3,
    min_labeled_neighbors: int = 2,
    low_height_radius: float = 0.2,
    low_height_threshold: float = 2.0,
) -> GSplatData:
    """
    幹ラベルを下から上へ層ごとに3D近傍で伝播する。
    林冠部の枝の連続性を保ちながらIDを割り当てる。

    Parameters
    ----------
    radius : 3D近傍探索の最大距離 [m]
    z_step : 層の厚み [m]
    z_window : 各層で参照する下方向の幅 [m]
    seed_radius : 幹シードの初期XY割り当て半径 [m]
    min_labeled_neighbors : 割り当てに必要な最小ラベル済み近傍数
    low_height_radius : 低高度層で使う半径 [m]（柵・ノイズへの拡散抑制）
    low_height_threshold : z_min からこの高さ[m]未満の層を低高度とみなす
    """
    if len(trunk_gs.centers) == 0:
        raise ValueError("trunk_gs is empty")

    full_gs.labels[:] = -1

    # --- シード初期化: 幹点のXY近傍でラベルを付与 ---
    seed_tree = cKDTree(trunk_gs.centers[:, :2])
    dist, ind = seed_tree.query(full_gs.centers[:, :2], workers=-1)
    seed_mask = dist < seed_radius
    full_gs.labels[seed_mask] = trunk_gs.labels[ind[seed_mask]]

    # --- Bottom-up 層ごとの伝播 ---
    z_vals = full_gs.centers[:, 2]
    z_min = z_vals.min()
    z_max = z_vals.max()

    for z_bottom in np.arange(z_min, z_max, z_step):
        z_top = z_bottom + z_step

        # 低高度層は半径を絞る（柵・ノイズへの拡散を抑制）
        current_height = z_bottom - z_min
        effective_radius = low_height_radius if current_height < low_height_threshold else radius

        layer_mask = (z_vals >= z_bottom) & (z_vals < z_top) & (full_gs.labels == -1)
        if not layer_mask.any():
            continue

        ref_mask = (full_gs.labels != -1) & (z_vals >= z_bottom - z_window) & (z_vals < z_top)
        if not ref_mask.any():
            continue

        ref_points = full_gs.centers[ref_mask]
        ref_labels = full_gs.labels[ref_mask]

        k = min(min_labeled_neighbors, len(ref_points))
        ref_tree = cKDTree(ref_points)
        dist, ind = ref_tree.query(
            full_gs.centers[layer_mask], k=k,
            distance_upper_bound=effective_radius, workers=-1,
        )

        if k == 1:
            dist = dist[:, np.newaxis]
            ind = ind[:, np.newaxis]

        within = dist < effective_radius  # (N_layer, k)
        enough = within.sum(axis=1) >= min_labeled_neighbors

        assign = within[:, 0] & enough
        layer_indices = np.where(layer_mask)[0]
        full_gs.labels[layer_indices[assign]] = ref_labels[ind[assign, 0]]

    return full_gs


def _voronoi_correction(
    full_gs: GSplatData,
    trunk_gs: GSplatData,
    max_centroid_dist: float = 3.0,
) -> GSplatData:
    """
    各ラベル済み点を幹重心への最近傍（ボロノイ）で再割り当てする。
    bottom-up伝播の到達順序による誤割り当てを補正し、
    競合領域を幹重心で自然に2分割する。
    幹重心から max_centroid_dist [m] 以上離れた点は -1 に戻す。
    """
    unique_labels = np.unique(trunk_gs.labels)
    unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) < 2:
        return full_gs

    centroids = np.array([
        trunk_gs.centers[trunk_gs.labels == l, :2].mean(axis=0)
        for l in unique_labels
    ])

    labeled_mask = full_gs.labels != -1
    if not labeled_mask.any():
        return full_gs

    centroid_tree = cKDTree(centroids)
    dist, ind = centroid_tree.query(full_gs.centers[labeled_mask, :2], workers=-1)

    labeled_indices = np.where(labeled_mask)[0]
    within = dist < max_centroid_dist
    full_gs.labels[labeled_indices[within]] = unique_labels[ind[within]]
    full_gs.labels[labeled_indices[~within]] = -1

    return full_gs


def _filter_by_noise_proximity(
    full_gs: GSplatData,
    trunk_gs: GSplatData,
    noise_proximity_radius: float = 0.15,
    seed_protection_radius: float = 0.35,
) -> GSplatData:
    """
    幹スライスのノイズ点（-1）XY位置と幹シードXY位置を用いて誤ラベルを除去する。

    「ノイズ点に近い かつ 幹シードから遠い」ラベル済み点を -1 に戻す。
    幹シードに近い点（木の根本）は保護される。
    HAGに依存しないため、柵に限らず幹高付近の非木構造物に汎用的に適用できる。

    Parameters
    ----------
    noise_proximity_radius : trunk_sliceのノイズ点からこのXY距離[m]以内を除去候補とする
    seed_protection_radius : 幹シードからこのXY距離[m]以内なら除去しない
    """
    noise_mask = trunk_gs.labels == -1
    seed_mask = trunk_gs.labels > 0
    if not noise_mask.any() or not seed_mask.any():
        return full_gs

    noise_tree = cKDTree(trunk_gs.centers[noise_mask, :2])
    seed_tree = cKDTree(trunk_gs.centers[seed_mask, :2])

    target_mask = full_gs.labels != -1
    if not target_mask.any():
        return full_gs

    target_xy = full_gs.centers[target_mask, :2]
    dist_noise, _ = noise_tree.query(target_xy, workers=-1)
    dist_seed, _ = seed_tree.query(target_xy, workers=-1)

    remove = (dist_noise < noise_proximity_radius) & (dist_seed > seed_protection_radius)
    target_indices = np.where(target_mask)[0]
    full_gs.labels[target_indices[remove]] = -1

    return full_gs


def _remove_small_components(
    full_gs: GSplatData,
    connect_radius: float = 0.6,
    min_component_size: int = 50,
) -> GSplatData:
    """
    ラベルごとに3D連結成分を計算し、小さい孤立クラスタを -1 に戻す。

    Parameters
    ----------
    connect_radius     : この距離[m]以内の点を連結とみなす
    min_component_size : これ未満の点数のクラスタを除去する
    """
    unique_labels = np.unique(full_gs.labels)
    unique_labels = unique_labels[unique_labels != -1]

    for label in unique_labels:
        mask = full_gs.labels == label
        points = full_gs.centers[mask]
        n = len(points)
        if n == 0:
            continue

        # 近傍ペアを列挙して疎行列を構築
        tree = cKDTree(points)
        pairs = tree.query_pairs(connect_radius, output_type='ndarray')

        if len(pairs) == 0:
            # 孤立点のみ → 全て除去
            full_gs.labels[mask] = -1
            continue

        row = np.concatenate([pairs[:, 0], pairs[:, 1]])
        col = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(row), dtype=np.int8)
        adj = csr_matrix((data, (row, col)), shape=(n, n))

        n_components, comp_labels = connected_components(adj, directed=False)

        # 小さい連結成分を除去
        indices = np.where(mask)[0]
        for c in range(n_components):
            comp_mask = comp_labels == c
            if comp_mask.sum() < min_component_size:
                full_gs.labels[indices[comp_mask]] = -1

    return full_gs


# ===============================
# 木の高さ計算
# ===============================
def compute_tree_heights(full_gs: GSplatData) -> dict[int, float]:

    heights = {}
    unique_labels = np.unique(full_gs.labels)

    for label in unique_labels:
        if label == -1:
            continue

        mask = full_gs.labels == label
        z_values = full_gs.centers[mask][:, 2]

        height = np.max(z_values) - np.min(z_values)
        heights[label] = height

    return heights


# ===============================
# 胸高直径計算
# ===============================
def compute_dbh(
    full_gs: GSplatData,
    ground_z: float,
    lower: float = 1.2,
    upper: float = 1.4,
) -> dict[int, float]:

    dbh_dict = {}
    unique_labels = np.unique(full_gs.labels)

    for label in unique_labels:
        if label == -1:
            continue

        mask = full_gs.labels == label
        points = full_gs.centers[mask]

        # 地面基準で胸高スライス
        slice_mask = (
            (points[:, 2] > ground_z + lower) &
            (points[:, 2] < ground_z + upper)
        )

        slice_points = points[slice_mask]

        if len(slice_points) < 30:
            continue
        slice_points = slice_points[:,:3]

        # RANSAC円フィット
        circle = pyrsc.Circle()
        center, _,radius,inliers = circle.fit(slice_points, thresh=0.02)

        dbh = radius * 2.0
        dbh_dict[label] = dbh

    return dbh_dict


# ===============================
# 全体実行パイプライン
# ===============================
def run_pipeline(
    full_gs: GSplatData,
    trunk_slice_gs: GSplatData,
    ground_z: float,
):

    # 1. 幹抽出
    trunk_classifier = TrunkClassifier(trunk_slice_gs)
    trunk_gs = trunk_classifier.run()

    # trunkのみ抽出
    trunk_mask = trunk_gs.labels != -1
    trunk_only = GSplatData(
        centers=trunk_gs.centers[trunk_mask],
        rgbs=trunk_gs.rgbs[trunk_mask],
        opacities=trunk_gs.opacities[trunk_mask],
        covariances=trunk_gs.covariances[trunk_mask],
        labels=trunk_gs.labels[trunk_mask],
    )

    # 2. 幹ID伝播（Bottom-up Region Growing）
    full_gs = propagate_trunk_ids_bottom_up(full_gs, trunk_only)

    # 2.5 ノイズ近傍フィルタ: 幹スライスのノイズXYに近く幹シードから遠い点を除去
    full_gs = _filter_by_noise_proximity(full_gs, trunk_gs)

    # 3. 高さ
    heights = compute_tree_heights(full_gs)

    # 4. DBH
    dbh = compute_dbh(full_gs, ground_z)

    return full_gs, heights, dbh