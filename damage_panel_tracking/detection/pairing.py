from __future__ import annotations

from typing import List, Tuple

from .types import PairMeta, xywh_to_xyxy


def horiz_overlap_ratio(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1, _, w1, _ = b1
    x2, _, w2, _ = b2
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    overlap = max(0, right - left)
    denom = float(min(w1, w2))
    return 0.0 if denom <= 0 else overlap / denom


def pair_boxes_same_color(
    boxes_xywh: List[Tuple[int, int, int, int]],
    width_tol: float,
    min_h_overlap: float,
    min_v_gap: int,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    boxes_sorted = sorted(boxes_xywh, key=lambda b: b[1])  # y asc
    paired: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
    used_bottom = set()

    for i, top in enumerate(boxes_sorted):
        x_t, y_t, w_t, h_t = top
        best_j = -1
        best_dy = None
        for j, bottom in enumerate(boxes_sorted):
            if j == i or j in used_bottom:
                continue
            x_b, y_b, w_b, h_b = bottom
            dy = y_b - (y_t + h_t)
            if dy < min_v_gap:
                continue
            if abs(w_t - w_b) > width_tol * max(w_t, w_b):
                continue
            if horiz_overlap_ratio(top, bottom) < min_h_overlap:
                continue
            if best_dy is None or dy < best_dy:
                best_dy = dy
                best_j = j
        if best_j >= 0:
            paired.append((top, boxes_sorted[best_j]))
            used_bottom.add(best_j)

    return paired


def union_bbox(top: Tuple[int, int, int, int], bottom: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, w1, h1 = top
    x2, y2, w2, h2 = bottom
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))


def build_pair_meta(
    color: str,
    top: Tuple[int, int, int, int],
    bottom: Tuple[int, int, int, int],
) -> PairMeta:
    u = union_bbox(top, bottom)
    return PairMeta(
        color=color,  # type: ignore
        top_xywh=top,
        bottom_xywh=bottom,
        union_xywh=u,
        union_xyxy=xywh_to_xyxy(u),
    )
