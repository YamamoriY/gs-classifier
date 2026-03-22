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


#ボロノイ的な伝播で精度が出ない。
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
    cross_penalty: float = 3.0,
    exp_scale: float = 10.0,
) -> GSplatData:
    """
    ミニマックスDijkstra + 局所PCA主軸アライメントによる幹ID伝播。

    各点は「シードから到達するパス上の最大エッジコストが最小」となるシードのラベルを得る。

    エッジコスト = exp(exp_scale * d) + cross_penalty * linearity * (1 - alignment)
    - exp(exp_scale * d): 距離の非線形スケーリング。構造間ギャップを強調
    - cross_penalty: エッジ方向がnbの主軸と直交するほどペナルティ

    Parameters
    ----------
    connect_radius : スケール済み座標空間での最大エッジ長 [m]
    k_neighbors    : 各点が参照する近傍数
    seed_radius    : 幹シードの初期3D割り当て半径 [m]（実空間）
    min_hag        : この高さ[m]未満の点には伝播しない
    xy_scale       : XY方向のペナルティ係数（縦方向バイアス）
    cross_penalty  : 主軸と直交する方向への伝播ペナルティ重み
    exp_scale      : 距離の指数スケーリング係数
    """
    if len(trunk_gs.centers) == 0:
        raise ValueError("trunk_gs is empty")

    N = len(full_gs.centers)
    centers = full_gs.centers  # (N, 3) 実空間座標
    full_gs.labels[:] = -1

    # --- シード初期化: 幹点の3D近傍でラベルを付与 ---
    seed_tree = cKDTree(trunk_gs.centers)
    dist, ind = seed_tree.query(centers, workers=-1)
    seed_mask = dist < seed_radius
    full_gs.labels[seed_mask] = trunk_gs.labels[ind[seed_mask]]

    # --- 有効点マスク ---
    valid = full_gs.additional_data["hags"] >= min_hag

    # --- XYスケールした座標空間でk-NNグラフ構築 ---
    scaled_centers = centers.copy()
    scaled_centers[:, 0] *= xy_scale
    scaled_centers[:, 1] *= xy_scale

    pt_tree = cKDTree(scaled_centers)
    dists, inds = pt_tree.query(scaled_centers, k=k_neighbors + 1, workers=-1)
    neighbor_inds = inds[:, 1:]
    neighbor_dists = dists[:, 1:]

    # --- 局所PCA主軸（実空間のk-NN近傍から計算） ---
    nb_pts = centers[neighbor_inds]  # (N, k, 3)
    means = nb_pts.mean(axis=1, keepdims=True)
    centered = nb_pts - means
    k = neighbor_inds.shape[1]
    covs = np.einsum('nki,nkj->nij', centered, centered) / max(k - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(covs)
    principal_axes = eigvecs[:, :, -1]  # (N, 3) 最大固有値の固有ベクトル
    linearity = np.where(
        eigvals[:, -1] > 1e-8,
        (eigvals[:, -1] - eigvals[:, -2]) / eigvals[:, -1],
        0.0,
    )  # 0=球状(針葉樹葉), 1=棒状(枝)

    # --- エッジコストの距離項を事前計算 ---
    exp_dists = np.exp(exp_scale * neighbor_dists)  # (N, k)

    # --- ミニマックスDijkstra ---
    cost = np.full(N, np.inf)
    parent = np.full(N, -1, dtype=int)
    labels = full_gs.labels  # 参照を保持

    heap = []  # (cost, idx)
    for i in np.where(seed_mask & valid)[0]:
        cost[i] = 0.0
        heapq.heappush(heap, (0.0, int(i)))

    while heap:
        c, idx = heapq.heappop(heap)
        if c > cost[idx]:
            continue
        lbl = labels[idx]
        idx_nbs = neighbor_inds[idx]
        idx_dists = neighbor_dists[idx]
        idx_exp = exp_dists[idx]
        for j in range(k):
            d = idx_dists[j]
            if d > connect_radius:
                break
            nb = idx_nbs[j]
            if not valid[nb]:
                continue

            # 主軸アライメント: エッジ方向とnbの主軸の一致度
            edge_vec = centers[nb] - centers[idx]
            edge_len = np.linalg.norm(edge_vec)
            if edge_len > 1e-8:
                alignment = abs(np.dot(principal_axes[nb], edge_vec / edge_len))
            else:
                alignment = 1.0

            edge_cost = idx_exp[j] + cross_penalty * linearity[nb] * (1.0 - alignment)
            new_cost = max(c, edge_cost)
            if new_cost < cost[nb]:
                cost[nb] = new_cost
                labels[nb] = lbl
                parent[nb] = idx
                heapq.heappush(heap, (new_cost, int(nb)))

    return full_gs, cost, parent, principal_axes, eigvals


#枝が途中で別の木に分類されちゃう境界を見つける。これ自体はうまくいったけどそっから活かせなかった。
def fix_absorbed_points(
    full_gs: GSplatData,
    parent: np.ndarray,
    principal_axes: np.ndarray,
    eigvals: np.ndarray,
    k_neighbors: int = 20,
    linearity_threshold: float = 0.5,
) -> np.ndarray:
    """
    枝に沿ってラベルが切り替わる点を検出する。

    各点のPCA主軸の正方向・負方向にある近傍を調べ、
    両側で異なるラベルを持つ場合、その点は枝上のラベル切り替え点。
    linearity が高い（棒状=枝）点のみ対象にすることで針葉樹の葉を除外。

    Parameters
    ----------
    principal_axes      : 各点の局所PCA主軸 (N, 3)
    eigvals             : 各点のPCA固有値 (N, 3)、昇順
    k_neighbors         : 近傍探索数
    linearity_threshold : この値以上のlinearityを持つ点のみ対象
    """
    N = len(full_gs.centers)
    centers = full_gs.centers
    labels = full_gs.labels.copy()

    # linearity: 棒状かどうか
    linearity = np.where(
        eigvals[:, -1] > 1e-8,
        (eigvals[:, -1] - eigvals[:, -2]) / eigvals[:, -1],
        0.0,
    )

    # k-NN
    tree = cKDTree(centers)
    _, nb_inds = tree.query(centers, k=k_neighbors + 1, workers=-1)
    nb_inds = nb_inds[:, 1:]

    # 親方向を事前計算
    has_parent = (parent >= 0) & (parent != np.arange(N))
    parent_dir = np.zeros((N, 3))
    parent_dir[has_parent] = centers[parent[has_parent]] - centers[has_parent]
    norms = np.linalg.norm(parent_dir, axis=1, keepdims=True)
    parent_dir /= np.maximum(norms, 1e-8)

    # 各点について、主軸の正方向・負方向の近傍ラベルを集計し、
    # 境界点では両側のparent方向の一貫性で正しい側を判定
    n_suspicious = 0
    n_changed = 0
    for i in range(N):
        if labels[i] == -1 or linearity[i] < linearity_threshold:
            continue
        axis = principal_axes[i]
        pos_nbs = []
        neg_nbs = []
        for nb in nb_inds[i]:
            if labels[nb] == -1:
                continue
            diff = centers[nb] - centers[i]
            dot = np.dot(axis, diff)
            if dot > 0:
                pos_nbs.append(nb)
            else:
                neg_nbs.append(nb)
        if not pos_nbs or not neg_nbs:
            continue
        pos_labels = [labels[nb] for nb in pos_nbs]
        neg_labels = [labels[nb] for nb in neg_nbs]
        pos_majority = np.bincount(pos_labels).argmax()
        neg_majority = np.bincount(neg_labels).argmax()
        if pos_majority == neg_majority:
            continue

        n_suspicious += 1

        # 両側のparent方向と枝方向（主軸）の一貫性を比較
        # 正しい側: parent方向が枝方向に沿ってる（alignmentが高い）
        # 吸われた側: parent方向が枝から逸れて幹に向かう（alignmentが低い）
        pos_align = np.mean([abs(np.dot(principal_axes[nb], parent_dir[nb]))
                            for nb in pos_nbs if has_parent[nb]] or [0.0])
        neg_align = np.mean([abs(np.dot(principal_axes[nb], parent_dir[nb]))
                            for nb in neg_nbs if has_parent[nb]] or [0.0])

        # alignment が高い側のラベルが正しい
        if pos_align >= neg_align:
            correct_label = pos_majority
        else:
            correct_label = neg_majority

        if labels[i] != correct_label:
            labels[i] = correct_label
            n_changed += 1

    print(f"\n=== 枝上ラベル切り替え検出 ===")
    print(f"  境界点: {n_suspicious} ({n_suspicious/N*100:.1f}%)")
    print(f"  ラベル変更: {n_changed} ({n_changed/N*100:.1f}%)")

    return labels


#ボトムアップだと，水平方向に広範囲に広がる枝に追従できないから，完璧な樹形を得られないのが問題．不採用
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
# Graph Cut によるラベル境界最適化
# ===============================
def refine_labels_graphcut(
    full_gs: GSplatData,
    cost: np.ndarray,
    neighbor_inds: np.ndarray,
    neighbor_dists: np.ndarray,
    smoothness_weight: float = 1.0,
) -> GSplatData:
    """
    ミニマックスの結果をGraph Cut (alpha-expansion) で最適化する。

    隣接する2ラベル間でバイナリGraph Cutを反復し、
    グローバルに最適なラベル境界を求める。

    データ項: ミニマックスコスト（低い=現ラベルへの確信が高い）
    平滑化項: 隣接点が異なるラベルを持つペナルティ（エッジ距離の逆数で重み付け）
    """
    import maxflow

    labels = full_gs.labels.copy()
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) < 2:
        return full_gs

    # alpha-expansion: 各ラベルについて順番にバイナリ最適化を繰り返す
    changed = True
    iteration = 0
    max_iterations = 5

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        for alpha in unique_labels:
            # alpha以外のラベルを持つ点のうち、alphaの近傍にある点が対象
            # 効率のため、alphaとの境界付近の点だけ処理する
            alpha_mask = labels == alpha
            if not alpha_mask.any():
                continue

            # alphaの近傍にある他ラベルの点を特定
            boundary_points = set()
            alpha_indices = np.where(alpha_mask)[0]
            for idx in alpha_indices:
                for nb in neighbor_inds[idx]:
                    if labels[nb] != alpha and labels[nb] != -1:
                        boundary_points.add(int(nb))
            # alpha自身の境界付近も含める
            for idx in list(boundary_points):
                for nb in neighbor_inds[idx]:
                    if labels[nb] != -1:
                        boundary_points.add(int(nb))
                        if labels[nb] == alpha:
                            # alphaの境界付近の点も含める
                            for nb2 in neighbor_inds[nb]:
                                if labels[nb2] != -1:
                                    boundary_points.add(int(nb2))

            # alpha + 境界付近の点集合
            candidate_indices = np.array(sorted(
                set(alpha_indices.tolist()) | boundary_points
            ))

            if len(candidate_indices) < 2:
                continue

            # ローカルインデックスへのマッピング
            n_local = len(candidate_indices)
            global_to_local = {g: l for l, g in enumerate(candidate_indices)}

            # Graph構築
            g = maxflow.Graph[float](n_local, n_local * 10)
            nodes = g.add_nodes(n_local)

            # データ項: 正規化ミニマックスコスト + 近傍ラベル一致率
            # cost低い → 確信高い → 変更しにくい
            # 近傍に同じラベルが多い → 変更しにくい
            candidate_costs = cost[candidate_indices]
            finite_mask = np.isfinite(candidate_costs)
            if finite_mask.any():
                c_min = candidate_costs[finite_mask].min()
                c_max = candidate_costs[finite_mask].max()
                cost_norm = np.where(
                    finite_mask,
                    (candidate_costs - c_min) / (c_max - c_min + 1e-8),
                    1.0,
                )
            else:
                cost_norm = np.ones(n_local)

            data_weight = 5.0
            for l_idx in range(n_local):
                g_idx = candidate_indices[l_idx]
                current_label = labels[g_idx]

                # 確信度: cost低い=確信高い(1に近い), cost高い=確信低い(0に近い)
                confidence = 1.0 - cost_norm[l_idx]

                if current_label == alpha:
                    # 現在alpha: 維持コスト=0, 変更コスト=確信度に比例
                    g.add_tedge(nodes[l_idx], data_weight * confidence, 0.0)
                else:
                    # 現在alpha以外: 維持コスト=確信度に比例, 変更コスト=0
                    g.add_tedge(nodes[l_idx], 0.0, data_weight * confidence)

            # 平滑化項: 隣接点が異なるラベルになるペナルティ
            added_edges = set()
            for l_idx in range(n_local):
                g_idx = candidate_indices[l_idx]
                for nb, d in zip(neighbor_inds[g_idx], neighbor_dists[g_idx]):
                    nb = int(nb)
                    if nb not in global_to_local:
                        continue
                    l_nb = global_to_local[nb]
                    edge_key = (min(l_idx, l_nb), max(l_idx, l_nb))
                    if edge_key in added_edges:
                        continue
                    added_edges.add(edge_key)

                    # 隣接点が異なるラベルを持つペナルティ（定数重み）
                    w = smoothness_weight
                    g.add_edge(nodes[l_idx], nodes[l_nb], w, w)

            # Min-cut実行
            g.maxflow()

            # 結果を反映: source側 = alpha以外, sink側 = alpha
            n_changed = 0
            for l_idx in range(n_local):
                g_idx = candidate_indices[l_idx]
                if g.get_segment(nodes[l_idx]) == 1:  # sink = alpha
                    if labels[g_idx] != alpha:
                        labels[g_idx] = alpha
                        n_changed += 1
                else:  # source = not alpha
                    if labels[g_idx] == alpha:
                        # alphaから別ラベルに変更 — 最近傍の非alphaラベルを付与
                        for nb in neighbor_inds[g_idx]:
                            if labels[nb] != alpha and labels[nb] != -1:
                                labels[g_idx] = labels[nb]
                                n_changed += 1
                                break

            if n_changed > 0:
                changed = True

        print(f"  Graph Cut iteration {iteration}: changed={changed}")

    total_changed = (full_gs.labels != labels).sum()
    print(f"\n=== Graph Cut ラベル最適化 ===")
    print(f"  反復回数: {iteration}")
    print(f"  総変更点数: {total_changed} ({total_changed/len(labels)*100:.1f}%)")

    full_gs.labels = labels
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

    # 2. 幹ID伝播（ミニマックスDijkstra + PCAアライメント）
    full_gs, *_ = propagate_trunk_ids_minimax(full_gs, trunk_only, xy_scale=3.5)

    # 2.7 孤立小クラスタ除去
    full_gs = _remove_small_components(full_gs)

    # 3. 高さ
    heights = compute_tree_heights(full_gs)

    # 4. DBH
    dbh_results = compute_dbh(trunk_only)

    return full_gs, heights, dbh_results