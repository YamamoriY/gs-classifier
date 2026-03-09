from __future__ import annotations
import numpy as np
from collections import deque
import heapq
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN

from src.lib.types.gstype import GSplatData
from src.lib.trunk.trunk import TrunkClassifier


# ===============================
# 幹ID伝播（同時多源BFS）
# ===============================
def propagate_trunk_ids_bfs(
    full_gs: GSplatData,
    trunk_gs: GSplatData,
    connect_radius: float = 0.5,
    k_neighbors: int = 20,
    seed_radius: float = 0.3,
    min_hag: float = 0.2,
    xy_scale: float = 3.0,
) -> GSplatData:
    """
    幹シードから同時多源BFSで全点にラベルを伝播する。
    複数の幹が同時に展開するため、各点は物理的に最短で到達できる幹のラベルを得る。
    隣接木の境界は自然にグラフ空間上のVoronoi分割になる。

    XY 方向を xy_scale 倍にスケールした座標空間でk-NNグラフを構築することで、
    水平方向より垂直方向（上方）への伝播を優先させる。
    実効距離: d_eff = sqrt(xy_scale²*(dx²+dy²) + dz²)

    Parameters
    ----------
    connect_radius : スケール済み座標空間での連結距離 [m]
    k_neighbors    : 各点が参照する近傍数
    seed_radius    : 幹シードの初期3D割り当て半径 [m]（スケール前の実空間）
    min_hag        : この高さ[m]未満の点には伝播しない
    xy_scale       : XY方向のペナルティ係数（大きいほど横方向に広がりにくい）
    """
    if len(trunk_gs.centers) == 0:
        raise ValueError("trunk_gs is empty")

    full_gs.labels[:] = -1

    # --- シード初期化: 幹点の3D近傍でラベルを付与 ---
    seed_tree = cKDTree(trunk_gs.centers)
    dist, ind = seed_tree.query(full_gs.centers, workers=-1)
    seed_mask = dist < seed_radius
    full_gs.labels[seed_mask] = trunk_gs.labels[ind[seed_mask]]

    # --- 有効点マスク（HAG下限で地面付近への漏れを抑制） ---
    valid = full_gs.additional_data["hags"] >= min_hag

    # --- XYをスケールして縦方向バイアスを付与した座標空間でk-NNグラフ構築 ---
    # d_eff = sqrt(xy_scale²*(dx²+dy²) + dz²) となり、水平方向が遠く見える
    scaled_centers = full_gs.centers.copy()
    scaled_centers[:, 0] *= xy_scale
    scaled_centers[:, 1] *= xy_scale

    pt_tree = cKDTree(scaled_centers)
    dists, inds = pt_tree.query(scaled_centers, k=k_neighbors + 1, workers=-1)
    neighbor_inds = inds[:, 1:]
    neighbor_dists = dists[:, 1:]

    # --- 同時多源BFS ---
    queue = deque(int(i) for i in np.where(seed_mask & valid)[0])
    while queue:
        idx = queue.popleft()
        lbl = full_gs.labels[idx]
        for nb, d in zip(neighbor_inds[idx], neighbor_dists[idx]):
            if d > connect_radius:
                break  # 距離昇順なので以降も範囲外
            if full_gs.labels[nb] == -1 and valid[nb]:
                full_gs.labels[nb] = lbl
                queue.append(nb)

    return full_gs


# ===============================
# 幹ID伝播（ミニマックスDijkstra）
# ===============================
def propagate_trunk_ids_minimax(
    full_gs: GSplatData,
    trunk_gs: GSplatData,
    connect_radius: float = 1.5,
    k_neighbors: int = 20,
    seed_radius: float = 0.3,
    min_hag: float = 0.2,
    xy_scale: float = 3.0,
) -> GSplatData:
    """
    ミニマックスDijkstra（ボトルネックパス）による幹ID伝播。

    各点は「シードから到達するパス上の最大エッジが最小」となるシードのラベルを得る。
    - 連続した枝: 全エッジが小さい（点間隔程度）→ ミニマックスコスト小 ✓
    - 不連続な跳躍: どこかに大きなエッジが必要 → ミニマックスコスト大 ✗

    BFSと違いホップ数ではなく「パスの連続性」で競合するため、
    ある木の枝に連続してつながっている点群はその木のラベルを保持しやすい。

    Parameters
    ----------
    connect_radius : スケール済み座標空間での最大エッジ長 [m]（BFSより大きくてよい）
    k_neighbors    : 各点が参照する近傍数
    seed_radius    : 幹シードの初期3D割り当て半径 [m]（実空間）
    min_hag        : この高さ[m]未満の点には伝播しない
    xy_scale       : XY方向のペナルティ係数（縦方向バイアス）
    """
    if len(trunk_gs.centers) == 0:
        raise ValueError("trunk_gs is empty")

    N = len(full_gs.centers)
    full_gs.labels[:] = -1

    # --- シード初期化: 幹点の3D近傍でラベルを付与 ---
    seed_tree = cKDTree(trunk_gs.centers)
    dist, ind = seed_tree.query(full_gs.centers, workers=-1)
    seed_mask = dist < seed_radius
    full_gs.labels[seed_mask] = trunk_gs.labels[ind[seed_mask]]

    # --- 有効点マスク ---
    valid = full_gs.additional_data["hags"] >= min_hag

    # --- XYスケールした座標空間でk-NNグラフ構築 ---
    scaled_centers = full_gs.centers.copy()
    scaled_centers[:, 0] *= xy_scale
    scaled_centers[:, 1] *= xy_scale

    pt_tree = cKDTree(scaled_centers)
    dists, inds = pt_tree.query(scaled_centers, k=k_neighbors + 1, workers=-1)
    neighbor_inds = inds[:, 1:]
    neighbor_dists = dists[:, 1:]

    # --- ミニマックスDijkstra ---
    # cost[i] = シードから i に至るパス上の最大エッジ長の最小値
    cost = np.full(N, np.inf)

    heap = []  # (cost, idx)
    for i in np.where(seed_mask & valid)[0]:
        cost[i] = 0.0
        heapq.heappush(heap, (0.0, int(i)))

    while heap:
        c, idx = heapq.heappop(heap)
        if c > cost[idx]:
            continue  # 古いエントリ
        lbl = full_gs.labels[idx]
        for nb, d in zip(neighbor_inds[idx], neighbor_dists[idx]):
            if d > connect_radius:
                break  # 距離昇順なので以降も範囲外
            if not valid[nb]:
                continue
            # ミニマックス: このパスでの最大エッジ = max(これまでの最大, 今のエッジ)
            new_cost = max(c, d)
            if new_cost < cost[nb]:
                cost[nb] = new_cost
                full_gs.labels[nb] = lbl
                heapq.heappush(heap, (new_cost, int(nb)))

    return full_gs


#ボトムアップだと，広範囲に広がる枝に追従できないから，完璧な樹形を得られないのが問題
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
    幹ラベルを下から上へ層ごとに3D近傍で伝播する（旧実装）。
    円柱的な形状を仮定しており、広葉樹には不向き。

    Parameters
    ----------
    radius : 3D近傍探索の最大距離 [m]
    z_step : 層の厚み [m]
    z_window : 各層で参照する下方向の幅 [m]
    seed_radius : 幹シードの初期XY割り当て半径 [m]
    min_labeled_neighbors : 割り当てに必要な最小ラベル済み近傍数
    low_height_radius : 低高度層で使う半径 [m]（柵・ノイズへの拡散抑制）
    low_height_threshold : HAG からこの高さ[m]未満の層を低高度とみなす
    """
    if len(trunk_gs.centers) == 0:
        raise ValueError("trunk_gs is empty")

    full_gs.labels[:] = -1

    # --- シード初期化: 幹点の3D近傍でラベルを付与 ---
    seed_tree = cKDTree(trunk_gs.centers)
    dist, ind = seed_tree.query(full_gs.centers, workers=-1)
    seed_mask = dist < seed_radius
    full_gs.labels[seed_mask] = trunk_gs.labels[ind[seed_mask]]

    # --- Bottom-up 層ごとの伝播（HAG基準） ---
    z_vals = full_gs.additional_data["hags"]
    z_min = z_vals.min()
    z_max = z_vals.max()

    for z_bottom in np.arange(z_min, z_max, z_step):
        z_top = z_bottom + z_step

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

        within = dist < effective_radius
        enough = within.sum(axis=1) >= min_labeled_neighbors

        assign = within[:, 0] & enough
        layer_indices = np.where(layer_mask)[0]
        full_gs.labels[layer_indices[assign]] = ref_labels[ind[assign, 0]]

    return full_gs


#結局伝播塗り替えちゃってあんまいい感じになんなかった．不採用．
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

#これも円柱にとどめちゃうからうまく樹形取れなかった．一旦不採用
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
from dataclasses import dataclass

@dataclass
class DBHResult:
    """胸高直径の計算結果"""
    dbh: float              # 胸高直径 [m]
    center: np.ndarray      # (2,) 円の中心XY座標
    radius: float           # 円の半径 [m]
    slice_xy: np.ndarray    # (N, 2) スライス点群のXY座標（クラスタリング後）

#円フィットの関数
def _fit_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """代数的最小二乗法による円フィット (Kasa法)"""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = result[0], result[1]
    radius = np.sqrt(result[2] + cx**2 + cy**2)
    return cx, cy, radius

#断面切り抜いた後円フィットする前一番幹っぽい部分を抽出できないかなという関数
def _largest_cluster(xy: np.ndarray, eps: float = 0.1, min_samples: int = 5) -> np.ndarray:
    """DBSCANで最大クラスタの点群を返す。クラスタが見つからない場合は元の点群をそのまま返す"""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) == 0:
        return xy
    return xy[labels == unique[np.argmax(counts)]]


def compute_dbh(
    trunk_gs: GSplatData,
    search_lower: float = 0.5,
    search_upper: float = 1.0,
    slice_width: float = 0.4,
    slice_step: float = 0.05,
    min_points: int = 10,
) -> dict[int, DBHResult]:

    results = {}
    unique_labels = np.unique(trunk_gs.labels)
    slice_starts = np.arange(search_lower, search_upper - slice_width + 1e-6, slice_step)

    for label in unique_labels:
        if label == -1:
            continue

        mask = trunk_gs.labels == label
        points = trunk_gs.centers[mask, :3]
        hags = trunk_gs.additional_data["hags"][mask]

        trunk_centroid = points[:, :2].mean(axis=0)
        best_result: DBHResult | None = None
        best_dist = np.inf

        for s in slice_starts:
            slice_mask = (hags > s) & (hags < s + slice_width)
            slice_points = points[slice_mask]

            if len(slice_points) < min_points:
                continue

            # DBSCANで最大クラスタを抽出してノイズ除去
            xy = _largest_cluster(slice_points[:, :2])

            if len(xy) < min_points:
                continue

            # クラスタ重心が幹中心に最も近いスライスを採用
            dist = np.linalg.norm(xy.mean(axis=0) - trunk_centroid)

            if dist < best_dist:
                best_dist = dist
                cx, cy, radius = _fit_circle(xy)
                best_result = DBHResult(
                    dbh=radius * 2.0,
                    center=np.array([cx, cy]),
                    radius=radius,
                    slice_xy=xy,
                )

        if best_result is not None:
            results[label] = best_result

    return results


# ===============================
# 全体実行パイプライン
# ===============================
def run_pipeline(
    full_gs: GSplatData,
    trunk_slice_gs: GSplatData,
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
        additional_data={k: v[trunk_mask] for k, v in trunk_gs.additional_data.items()},
    )

    # 2. 幹ID伝播（同時多源BFS）
    full_gs = propagate_trunk_ids_minimax(full_gs, trunk_only, xy_scale=3.5)

    # 2.7 孤立小クラスタ除去
    full_gs = _remove_small_components(full_gs)

    # 3. 高さ
    heights = compute_tree_heights(full_gs)

    # 4. DBH
    dbh_results = compute_dbh(trunk_only)

    return full_gs, heights, dbh_results