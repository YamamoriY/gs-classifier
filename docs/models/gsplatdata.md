# GSplatData

`src/gs_classifier/models/gsplat.py`

3D Gaussian Splatting のデータを格納する主要なデータ構造です。

## 基本属性

| 属性 | 型 | 説明 |
|---|---|---|
| `centers` | `(N, 3)` float | ガウシアンの中心座標 |
| `rgbs` | `(N, 3)` float | RGB 色情報 |
| `opacities` | `(N, 1)` float | 不透明度 |
| `covariances` | `(N, 3, 3)` float | 共分散行列 |
| `labels` | `(N,)` int | 分類ラベル（デフォルト: 全て 0） |
| `additional_data` | `dict[str, ndarray]` | 追加データ（例: `hags`） |

## 主要メソッド

### `split_by_label() -> list[GSplatData]`

ラベルごとに `GSplatData` を分割します。ラベルの小さい順にソートされます。
分割後の各 `GSplatData` の labels は 0 にリセットされます。

```python
data_list = gsplat_data.split_by_label()
```

### `save_to_npz(path) / load_from_npz(path)`

NPZ 形式での保存・読み込み。`additional_data` も自動的に保存されます。

```python
gsplat_data.save_to_npz("output.npz")
loaded = GSplatData.load_from_npz("output.npz")
```

### `coordinate_transform(R)`

3x3 回転行列 `R` で座標変換を行います。`centers` と `covariances` の両方が変換されます。

```python
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # x軸周り-90度
gsplat_data.coordinate_transform(R)
```

### `copy() -> GSplatData`

深いコピーを返します。`additional_data` もコピーされます。

### `reset_labels()`

すべてのラベルを 0 にリセットします。

### `concatenate(other) -> GSplatData`

2 つの `GSplatData` を結合した新しいオブジェクトを返します。

### `print_shape()`

デバッグ用。各属性の shape と値の範囲、ラベル分布を表示します。

## 追加データについて

`additional_data` には `(N,)` の配列を格納できます。

- `split_by_label()` で分割する際、長さが `centers` と一致する配列のみ継承されます
- `save_to_npz` / `load_from_npz` で自動的に永続化されます

```python
gsplat_data.additional_data["hags"] = hags_array
```
