# gs-classifier

## 概要

3D Gaussian Splatting (3DGS) のデータを処理し、森林の要素（地面、幹、葉など）を分類するツールです。

## プロジェクト構造

```
gs-classifier/
├── src/gs_classifier/
│   ├── cli.py         # CLI エントリポイント
│   ├── pipeline/      # 分類パイプライン
│   ├── core/          # コア処理モジュール
│   ├── models/        # データモデル定義
│   ├── api/           # 外部 API 連携
│   ├── util/          # ユーティリティ
│   └── viewer/        # 3D 可視化ビューアー
├── tests/             # テストコード
├── data/              # 入力データ（.ply ファイル）
└── tmp/               # 処理結果の保存先
```

詳細なモジュール構成は [docs/README.md](docs/README.md) を参照してください。

## セットアップ

```bash
uv sync
```

## CLI

すべての操作は `gs-classifier` コマンドから実行できます。

```bash
gs-classifier --help
```

### 分類パイプライン

森の 3DGS データから単木抽出まで行う手順です。
各ステップの結果は `tmp/` に `.npz` 形式で保存されます。

```bash
# 1. 地面抽出（PLY → ノイズ除去 → 地面分類）
gs-classifier ground --ply data/takino.ply

# 2. 幹抽出（地面抽出の結果から幹を検出）
gs-classifier trunk

# 3. 単木抽出（幹の情報から個々の木を分類し、out/ に PLY と JSON を出力）
gs-classifier tree
```

各ステップは前のステップの出力（`tmp/*.npz`）に依存するため、順番に実行してください。

#### オプション

| オプション | 説明 |
|---|---|
| `--ply PATH` | 入力 PLY ファイルのパス（`ground` のみ、デフォルト: `data/takino.ply`） |
| `--no-viewer` | 処理後のビューアー表示をスキップ |

```bash
# ビューアーを表示せずに全パイプラインを実行
gs-classifier ground --ply data/forest.ply --no-viewer
gs-classifier trunk --no-viewer
gs-classifier tree --no-viewer
```

### ビューアー

PLY ファイルや NPZ ファイルを 3D ビューアーで表示します。

```bash
# PLY ファイルを表示
gs-classifier view data/takino.ply

# 複数ファイルを同時に表示
gs-classifier view tmp/ground_gs.npz tmp/above_ground_gs.npz

# PLY と NPZ を混在して表示
gs-classifier view data/takino.ply tmp/mid_gs.npz
```

ブラウザで `http://localhost:8080` にアクセスして表示を確認できます。

### テストの実行

```bash
uv run pytest tests/ -v
```

## 依存関係

主要な依存パッケージ（`uv` で管理）：

- numpy, open3d, viser, scikit-learn, matplotlib, plyfile
- pyransac3d, scipy, laspy

## 詳細

各モジュール・アルゴリズムの詳細は [docs/](docs/README.md) を確認してください。
