# ビューアー

## 概要

3D Gaussian Splatting データを可視化するツールです。viser を使用しています。 

ガウシアン、点群、ラベルごとの色付きガウシアン、の表示を切り替えれるのがいいところですが、viser の仕様により内部実装が複雑になってしまっています。 \
（一度プロットしたガウシアンを後から変更できないので、最初に通常ガウシアンと、色付きガウシアンを同時に追加して切り替えて表示している）

## 基本的な使い方

### 初期化

```python
from src.viewer.viewer import Viewer

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

**パラメータ**:
- `gsplat_data`: 表示する `GSplatData`
- `name`: 表示名（必ず重複なし！！！）
- `folder_name`: フォルダ名（同じフォルダ名のものは同じフォルダに表示）
- `visible`: 初期表示状態
- `image_out`: 画像出力対象にするかどうか

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

ビューのUIで切り替えができます。

- **Normal**: 通常表示
- **Points View**: ポイント表示
- **Class View**: クラス表示

### 画像出力

`image_out=True` に設定したデータは、「Print Images」ボタンで自動的に画像を出力できます。

- 4方向（前後左右）から画像を撮影
- `tmp/images/{name}/` に保存

