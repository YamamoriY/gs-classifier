# 地面検出

`src/gs_classifier/core/ground/`

## 概要

3DGS データから地面を検出し、地面・地面下・地面上に分類します。

高レベル API: `GroundDetector` (`ground.py`)
低レベル API: `SegGround`, `GroundLerp` (`groundcore.py`)

```python
from gs_classifier.core.ground import GroundDetector

detector = GroundDetector(gs)
ground_gs, under_ground_gs, above_ground_gs = detector.run(
    under_threshold=0.1, above_threshold=0.2
)
```

## 手法

### 四分木による地面高さ推定 (`SegGround`)

四分木を使用して、フィールド全体の地面の高さを推定します。

1. **四分木の構築**: フィールドを再帰的に 4 分割し、各セルで地面の高さを推定
2. **高さ推定**: 各セル内で、円柱状の範囲内の点の高さ分布から最頻値（ヒストグラム）を取得
3. **補間**: 四分木の結果から `RegularGridInterpolator` で線形補間関数を作成

**パラメータ:**
- `depth`: 四分木の深さ（デフォルト: 6）。深いほど精度が上がるが計算量も増加

### 分類 (`GroundLerp`)

補間された地面の高さから、各点の HAG (Height Above Ground) を算出し、3 つに分類：

- **地面下**: HAG < -`under_threshold`
- **地面**: -`under_threshold` <= HAG < `above_threshold`
- **地面上**: HAG >= `above_threshold`

分類後、`additional_data["hags"]` に HAG が格納されます。
