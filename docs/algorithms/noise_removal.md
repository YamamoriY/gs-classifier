# ノイズ除去

`src/gs_classifier/core/denoise/`

## 概要

3DGS データからノイズを除去する処理です。

高レベル API: `NoiseRemover` (`denoise.py`)
低レベル API: `DenoiseCore` (`denoisecore.py`)

```python
from gs_classifier.core.denoise import NoiseRemover

remover = NoiseRemover(gs)
clean_gs, noise_gs = remover.run_2d_dbscan(radius=0.1)
```

## 手法

### 2 次元 DBSCAN (`run_2d_dbscan`)

2 次元平面 (XY) 上で DBSCAN クラスタリングを行い、最大のクラスタを抽出します。
主に 3DGS を扱う初期段階で使うノイズ除去です。

**パラメータ:**
- `radius`: DBSCAN の eps = radius * 2, min_samples = 50

### 3 次元密度フィルタ (`run_3d_density`)

密度の低い点を除去します。
幹抽出の前に、密度の低い葉を取り除くために使います。

**パラメータ:**
- `radius`: 探索半径
- `point_count`: この数未満の近傍点数ならノイズと判定

### 低レベル API (DenoiseCore)

`DenoiseCore` は `np.ndarray` を直接扱う低レベルなクラスです。`NoiseRemover` は `GSplatData` のラッパーとして `DenoiseCore` を利用しています。

追加のデノイズアルゴリズムは `denoisecore.py` に実装するか、新しいファイルを `core/denoise/` に追加してください。
