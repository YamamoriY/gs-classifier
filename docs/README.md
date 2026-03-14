# 開発者向けドキュメント

## 設計思想

### データ構造の統一

このプロジェクトでは、3D Gaussian Splatting のデータを `GSplatData` という統一されたデータ構造で扱います。すべての処理は `GSplatData` を受け取り、`GSplatData` を返すことで、処理の組み合わせが容易になります。

### ラベルベースの分類

`GSplatData` はガウシアンスプラッティングに必須な情報に加えて、分類結果を格納できます。分類結果は `GSplatData.labels` に格納されます。各ガウシアンには整数のラベルが割り当てられ、`split_by_label()` メソッドでラベルごとに分割できます。これにより、柔軟な分類処理が可能になります。

### モジュール化

各機能は独立したモジュールとして実装されています：

- **Detector**: 検出処理（例: `GroundDetector`, `LeafDetector`）
- **Classifier**: 分類処理（例: `TrunkClassifier`）
- **Core**: 低レベルな処理（例: `DenoiseCore`, `SegGround`）

高レベルな処理（Detector/Classifier）は、低レベルな処理（Core）をラップして、`GSplatData` を扱いやすくしています。

### 処理の流れ

典型的な処理の流れは以下の通りです：

1. PLY ファイルを読み込んで `GSplatData` に変換
2. ノイズ除去
3. 地面検出
4. 幹・葉などの分類
5. 結果を可視化または保存

各ステップは独立しているため、必要に応じて組み合わせることができます。

## モジュール構成

```
src/gs_classifier/
├── pipeline/                 # パイプライン実行スクリプト
│   ├── detect_ground.py      # 地面抽出パイプライン
│   ├── detect_trunk.py       # 幹抽出パイプライン
│   └── detect_tree.py        # 単木抽出パイプライン
├── core/                     # コア処理モジュール
│   ├── gsloader.py           # PLY / Splat ファイル I/O
│   ├── kdtree.py             # KDTree (円柱探索・高さ探索)
│   ├── denoise/              # ノイズ除去
│   │   ├── denoise.py        # NoiseRemover (GSplatData ラッパー)
│   │   └── denoisecore.py    # DenoiseCore (低レベルアルゴリズム)
│   ├── ground/               # 地面検出
│   │   ├── ground.py         # GroundDetector (GSplatData ラッパー)
│   │   └── groundcore.py     # SegGround, GroundLerp (四分木・補間)
│   ├── leaf/                 # 葉検出
│   │   ├── leaf.py           # LeafDetector (GSplatData ラッパー)
│   │   └── leafcore.py       # LeafClassifier (HSV ベース分類)
│   ├── trunk/                # 幹分類
│   │   └── trunk.py          # TrunkClassifier (DBSCAN ベース)
│   └── image2d/              # 2D セグメンテーション
│       └── image2d.py        # Image2D (KNN ベース木分類)
├── models/                   # データモデル
│   ├── gsplat.py             # GSplatData
│   └── trunk.py              # TrunkLocation
├── api/                      # 外部 API
│   └── plantnet.py           # PlantNet 樹種同定 API
├── util/                     # ユーティリティ
│   └── pcd.py                # 点群操作 (Open3D, LAS I/O)
└── viewer/                   # 可視化
    ├── viewer.py             # Viewer (viser ベース)
    ├── types.py              # GSplatHandle, GSplatFolder, GSplatMode
    └── colors.py             # ColorCycle (カラーパレット)
```

## インポート例

`__init__.py` により、主要クラスはパッケージから直接インポートできます：

```python
from gs_classifier.models import GSplatData, TrunkLocation
from gs_classifier.core import load_ply_file, save_ply_file, KDTree
from gs_classifier.core.denoise import NoiseRemover
from gs_classifier.core.ground import GroundDetector
from gs_classifier.core.leaf import LeafDetector
from gs_classifier.core.trunk import TrunkClassifier
from gs_classifier.core.image2d import Image2D
from gs_classifier.viewer import Viewer
```

## 拡張方法

### 新しい処理アルゴリズムを追加する場合

各 `core/` サブモジュールは独立したディレクトリになっているので、新しいアルゴリズムをファイル追加で拡張できます。

例: 新しいデノイズ手法を追加する場合

1. `core/denoise/` に新しいファイル（例: `statistical.py`）を作成
2. `core/denoise/__init__.py` に re-export を追加
3. `pipeline/` のスクリプトから利用

### 新しいパイプラインを追加する場合

`pipeline/` に新しい Python ファイルを追加し、`core/` のモジュールを組み合わせてください。

## ドキュメント構成

- [データモデル](models/) - `GSplatData`, `TrunkLocation` の詳細
- [アルゴリズム](algorithms/) - 各処理手法の詳細
- [ビューアー](viewer/) - 可視化ツールの使い方
