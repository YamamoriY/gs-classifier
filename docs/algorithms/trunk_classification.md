# 幹分類

`src/gs_classifier/core/trunk/`

## 概要

地面上のデータから幹を検出・分類します。
このクラスは地上高 1-2m 程度の切り出された範囲を受け取ることを想定しています。

```python
from gs_classifier.core.trunk import TrunkClassifier

classifier = TrunkClassifier(mid_height_gs)
result_gs = classifier.run()
```

## 処理の流れ

### 1. DBSCAN クラスタリング (`dbscan_trunk`)

高さ z 方向の距離の重みを 0.1 倍に小さくし、ほぼ 2 次元平面上で DBSCAN クラスタリングを行います。これにより、幹の候補を空間的にグループ化します。

- `eps=0.05`, `min_samples=50`
- ノイズ点はラベル `-1` が割り当てられます

### 2. 高さ範囲チェック (`is_trunk_check_range`)

高さ方向の範囲が 0.8m 未満のクラスタをノイズとして除去します。
地上 1-2m を切り出した場合、幹は必ず下から上まで詰まっているはずだからです。

### 3. Z 軸 DBSCAN (`is_trunk_dbscan`)

各クラスタに対して Z 軸のみで DBSCAN を実行し、クラスタ数が 1 つでないものをノイズとします。
これは z 軸方向に中空の領域があるものは幹ではないと判断するためです。

- `eps=0.15`, `min_samples=50`

2 と 3 を組み合わせることで、「高さ 1-2m あたりに縦長でずっと分布しているノイズ」以外は取り除けます。

## その他の手法（未使用）

以下の手法は実装済みですが、現在のパイプラインでは不採用です：

- `is_trunk_fit_line`: pyransac3d による直線フィット検証
- `is_trunk_std`: 標準偏差ベースの外れ値検出
