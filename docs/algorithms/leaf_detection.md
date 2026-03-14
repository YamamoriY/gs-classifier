# 葉検出

`src/gs_classifier/core/leaf/`

## 概要

地面上のデータから葉を HSV 色空間で検出します。

高レベル API: `LeafDetector` (`leaf.py`)
低レベル API: `LeafClassifier`, `rgb_to_hsv` (`leafcore.py`)

```python
from gs_classifier.core.leaf import LeafDetector

detector = LeafDetector(above_ground_gs)
leaf_gs, objects_gs = detector.run()
```

## 手法

### HSV 色空間による分類 (`LeafClassifier`)

RGB 値にシグモイド関数を適用した後、HSV に変換し、以下の色範囲を葉と判定します：

**緑の葉:**
- 色相 (H): 0.17 - 0.67（緑〜青系）
- 彩度 (S): >= 0.02
- 明度 (V): >= 0.05

**黄色の葉（日光で黄色っぽくなったもの）:**
- 色相 (H): 0.08 - 0.17
- 彩度 (S): >= 0.1
- 明度 (V): >= 0.2

### 注意

- PLY ファイルの RGB 値はシグモイド適用前の状態で格納されているため、`rgb_to_hsv` 内でシグモイドを適用しています
- 単木抽出の前処理として使用することを想定しています
