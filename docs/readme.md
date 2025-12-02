# gs-classifier ドキュメント

## 概要

3D Gaussian Splatting (3DGS) のデータを処理し、森林の要素（地面、幹、葉など）を分類するツールです。

## プロジェクト構造

```
gs-classifier/
├── src/
│   ├── classifier/     # 分類処理の実行スクリプト
│   ├── lib/            # コアライブラリ
│   └── viewer/         # 3D可視化ビューアー
├── data/               # 入力データ（.plyファイル）
└── tmp/                # 処理結果の保存先
```

## 主要機能

### 分類処理

- **地面抽出** (`src/classifier/ground.py`)

  - ノイズ除去
  - 地面の検出と分類

- **幹抽出** (`src/classifier/trunk.py`)

  - 中央高度の抽出
  - 幹の分類

- **木抽出** (`src/classifier/tree.py`)
  - 幹の情報から木を抽出

### コアライブラリ (`src/lib/`)

- `gsloader.py`: PLY ファイルの読み込み
- `denoise/`: ノイズ除去
- `ground/`: 地面検出
- `trunk/`: 幹の分類
- `leaf/`: 葉の検出
- `types/gstype.py`: データ構造の定義

### ビューアー (`src/viewer/`)

- 3D 可視化（viser 使用）
- 分類結果の表示
- 画像出力機能

## 使用方法

### 実行例

```bash
# ビューアーの起動
python -m src.viewer.viewer

# 地面の分類
python -m src.classifier.ground
```

## 依存関係

主要な依存パッケージ：

- numpy
- open3d
- viser
- scikit-learn
- matplotlib
- plyfile

詳細は `requirements.txt` または `environment.yml` を参照してください。
