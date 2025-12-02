# gs-classifier

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

### 分類の実行例

実行結果は tmp/ に .npz 形式で格納されます。これは src/lib/types/gstype.py の GSplatData のオブジェクトと等価です。

```bash
# 1. data/ に 森の3dgsの .ply データを入れる
# （デフォルトだと takino.ply かな）

# 2. 地面抽出
python -m src.classifier.ground

# 3. 幹抽出
python -m src.classifier.trunk

# 4. 単木抽出
python -m src.classifier.tree
```

### ビューアーを実行する
```bash
# ビューアーを単独で実行する
python -m src.viewer.viewer
```



## 依存関係

主要な依存パッケージ：

- numpy
- open3d
- viser
- scikit-learn
- matplotlib
- plyfile

pip で全部入れられます。諸事情で requirements.txt が今ないです。今度作ります

## 詳細
各クラスの詳細は docs/ を確認してください。
