# ビューアー

`src/gs_classifier/viewer/`

## 概要

3D Gaussian Splatting データを可視化するツールです。viser を使用しています。

ガウシアン、点群、ラベルごとの色付きガウシアンの表示を切り替えられます。
viser の仕様により、一度プロットしたガウシアンを後から変更できないため、最初に通常ガウシアンと色付きガウシアンを同時に追加して切り替えて表示しています。

## 基本的な使い方

### 初期化

```python
from gs_classifier.viewer import Viewer

viewer = Viewer()
```

### データの追加

```python
viewer.add_gsplat(
    gsplat_data,
    name="ground",
    folder_name="ground",
    visible=True,
    image_out=False
)
```

**パラメータ:**

| パラメータ | 型 | 説明 |
|---|---|---|
| `gsplat_data` | `GSplatData` | 表示するデータ |
| `name` | `str` | 表示名（重複不可） |
| `folder_name` | `str` | フォルダ名（同名はグループ化される） |
| `visible` | `bool` | 初期表示状態 |
| `image_out` | `bool` | 画像出力対象にするか |

### 実行

```python
viewer.run()
```

ブラウザで `http://localhost:8080` にアクセスして表示を確認できます。

## 機能

### フォルダ管理

同じ `folder_name` のデータは同じフォルダに表示され、以下の操作が可能です：

- **Show All / Hide All**: フォルダ内のすべてのデータを表示/非表示
- **< / >**: 前後のデータを順番に表示

### 表示モード

ビューの UI で切り替えができます。

- **Normal**: 通常のガウシアン表示
- **Points View**: ポイントクラウド表示
- **Class View**: ラベルごとに色分けされた表示

### 画像出力

`image_out=True` に設定したデータは、「Print Images」ボタンで自動的に画像を出力できます。

- 4 方向（前後左右）から画像を撮影
- `tmp/images/{name}/` に保存

## 内部構造

| ファイル | クラス | 説明 |
|---|---|---|
| `viewer.py` | `Viewer` | メインビューアー |
| `types.py` | `GSplatHandle` | 表示モード管理 (normal/class/points) |
| `types.py` | `GSplatFolder` | フォルダ管理 (show/hide/step) |
| `types.py` | `GSplatMode` | 表示モード Enum |
| `colors.py` | `ColorCycle` | シングルトンカラーパレット (tab20) |
