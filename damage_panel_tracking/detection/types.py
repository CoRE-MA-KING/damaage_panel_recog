from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np

ColorName = Literal["blue", "red"]


@dataclass(frozen=True)
class PairMeta:
    """同色の上下LEDペア1件を表す検出結果。

    座標表現（情報としては同一）:
    - xywh: (x, y, width, height)
    - xyxy: (x1, y1, x2, y2)
    """

    # ペアの色ラベル（blue/red）。
    color: ColorName
    # 上側LEDのbbox（xywh）。
    top_xywh: Tuple[int, int, int, int]
    # 下側LEDのbbox（xywh）。
    bottom_xywh: Tuple[int, int, int, int]
    # 上下LEDをまとめた外接bbox（xywh）。
    union_xywh: Tuple[int, int, int, int]
    # 上下LEDをまとめた外接bbox（xyxy）。
    union_xyxy: np.ndarray  # 浮動小数のbbox座標 [x1,y1,x2,y2]


def xywh_to_xyxy(box_xywh: Tuple[int, int, int, int]) -> np.ndarray:
    # OpenCV流のxywhをxyxy座標へ変換する。
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=float)


def xyxy_center(box_xyxy: np.ndarray) -> Tuple[float, float]:
    # bbox中心を浮動小数ピクセル座標で計算する。
    x1, y1, x2, y2 = box_xyxy.tolist()
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # 2つのxyxy bboxのIoUを計算する。
    ax1, ay1, ax2, ay2 = a.tolist()
    bx1, by1, bx2, by2 = b.tolist()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return inter / union
