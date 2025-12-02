# データ構造

## GSplatData

3D Gaussian Splatting のデータを格納する主要なデータ構造です。

### 基本属性

- `centers`: ガウシアンの中心座標 `(N, 3)`
- `rgbs`: RGB色情報 `(N, 3)`
- `opacities`: 不透明度 `(N, 1)`
- `covariances`: 共分散行列 `(N, 3, 3)`
- `labels`: 分類ラベル `(N,)` (整数配列)
- `additional_data`: 追加データ（辞書形式）

### 主要メソッド

#### `split_by_label()`

ラベルごとに `GSplatData` を分割します。

```python
# ラベル0とラベル1に分割
data_list = gsplat_data.split_by_label()
```

#### `save_to_npz(path)`

NPZ形式で保存します。

```python
gsplat_data.save_to_npz("output.npz")
```

#### `load_from_npz(path)`

NPZ形式から読み込みます。

```python
gsplat_data = GSplatData.load_from_npz("output.npz")
```

#### `coordinate_transform(R)`

座標変換を行います。

```python
# 回転行列Rで変換
gsplat_data.coordinate_transform(R)
```

#### `reset_labels()`

すべてのラベルを0にリセットします。

### 追加データ

`additional_data` には任意のデータを格納できます。例：

- `hags`: 地面からの高さ（Height Above Ground）

注意: `split_by_label()` で分割する際、`additional_data` の各配列の長さが `centers` の長さと一致する必要があります。

