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

1. PLYファイルを読み込んで `GSplatData` に変換
2. ノイズ除去
3. 地面検出
4. 幹・葉などの分類
5. 結果を可視化または保存

各ステップは独立しているため、必要に応じて組み合わせることができます。

## ドキュメント構成

- [アルゴリズム](algorithms/) - 各処理の手法の詳細
- [データ構造](types/) - `GSplatData` の詳細
- [ビューアー](viewer/) - 可視化ツールの使い方
