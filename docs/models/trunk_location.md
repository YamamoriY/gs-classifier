# TrunkLocation

`src/gs_classifier/models/trunk.py`

幹の位置情報を格納するデータ構造です。

## 属性

| 属性 | 型 | 説明 |
|---|---|---|
| `trunk_locations` | `(N, 3)` float | 各幹の中心座標 |

## メソッド

### `from_gs(gs: GSplatData) -> TrunkLocation`

ラベル付き `GSplatData` から幹位置を算出します。各ラベルの `centers` の平均を幹の位置とします。ラベル `-1` は除外されます。

```python
trunk_loc = TrunkLocation.from_gs(labeled_gs)
```

### `coordinate_transform(R)`

3x3 回転行列で座標変換します。

### `save_to_json(path)`

JSON 形式で保存します。各幹は `name`, `x`, `y`, `z` のフィールドを持ちます。

```json
[
    {"name": "trunk_0", "x": 1.0, "y": 2.0, "z": 0.5},
    {"name": "trunk_1", "x": 5.0, "y": 3.0, "z": 0.4}
]
```
