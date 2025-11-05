import numpy as np
from src.lib.types import GSplatData


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    # シグモイドを作用
    rgb = 1 / (1 + np.exp(-rgb))

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    v = maxc
    
    deltac = maxc - minc
    s = np.where(maxc != 0, deltac / maxc, 0)
    
    # Hue の計算
    h = np.zeros_like(v)
    
    idx = (maxc == r)
    h[idx] = ((g[idx] - b[idx]) / deltac[idx]) % 6
    
    idx = (maxc == g)
    h[idx] = ((b[idx] - r[idx]) / deltac[idx]) + 2
    
    idx = (maxc == b)
    h[idx] = ((r[idx] - g[idx]) / deltac[idx]) + 4
    
    h[deltac == 0] = 0
    h = h / 6.0  # 0-1 に正規化
    
    return np.stack([h, s, v], axis=1)

# 雑に葉っぱを取り除くクラス
# NOTE: hsv空間で DBSCAN とかしたらもっとよくなりそう
class LeafClassifier:
    gs: GSplatData
    def __init__(self, gs: GSplatData):
        self.gs = gs

    def classify_leaf(self) -> GSplatData:
        hsv = rgb_to_hsv(self.gs.rgbs)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        
        # 緑・青系の色の条件
        # Hue: 緑は約60°-180°（0.17-0.5）、青は約180°-240°（0.5-0.67）の範囲
        # Saturation: ある程度彩度が高い（灰色っぽくない）
        # Value: 暗めの色も含めるため、明度の下限を下げる
        
        green_mask = (
            (h >= 0.17) & (h <= 0.67) &  # 緑・青系の色相
            (s >= 0.02) &                # 彩度条件を緩和
            (v >= 0.05)                  # 明度条件を緩和（暗めの色も含める）
        )
        
        # 黄色系の色の条件（彩度が高い場合）
        # Hue: 黄色は約50°-60°（0.14-0.17）の範囲
        # Saturation: 彩度が高い（鮮やかな黄色）
        yellow_mask = (
            (h >= 0.08) & (h <= 0.17) &  # 黄色系の色相
            (s >= 0.1) &                 # 高い彩度
            (v >= 0.2)                  # 最低限の明度
        )
        
        # 緑・青系または黄色系のマスクを結合
        leaf_mask = green_mask | yellow_mask
        leaf_indices = np.where(leaf_mask)[0]
        
        return leaf_indices
